from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from motion_analysis import (
    overlay_detections_and_save_color,
    overlay_detections_and_save_orb,
    overlay_dense_optical_flow_and_save,
    overlay_sparse_optical_flow_and_save,
    overlay_background_subtraction_and_save,
    overlay_frame_differencing_and_save
)
import tempfile
import os
import datetime
import subprocess

app = FastAPI()

# Ensure the 'detections' folder exists in the current directory
DETECTIONS_DIR = os.path.join(os.getcwd(), "detections")
os.makedirs(DETECTIONS_DIR, exist_ok=True)

# Serve the 'detections' folder as static files
app.mount("/detections", StaticFiles(directory=DETECTIONS_DIR), name="detections")

@app.post("/track_color")
async def track_color(video: UploadFile = File(...), object_image: UploadFile = File(...)):
    """
    Endpoint to track an object in a video using color-based detection and save the annotated video in 'detections' folder.
    """
    # Temporary files for input
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(await video.read())
        temp_video_path = temp_video.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_image:
        temp_image.write(await object_image.read())
        temp_image_path = temp_image.name

    # Generate a unique output filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_video_filename = f"color_detection_{timestamp}.mp4"
    output_video_path = os.path.join(DETECTIONS_DIR, output_video_filename)
    temp_output_path = output_video_path + ".tmp.mp4"  # Temporary file for OpenCV output

    try:
        # Process with save=True, get HSV bounds for debugging
        tracked_positions, lower_hsv, upper_hsv = overlay_detections_and_save_color(
            temp_video_path, temp_image_path, temp_output_path, save=True
        )

        # Convert the video to H.264/AAC MP4 using FFmpeg
        subprocess.run([
            "ffmpeg",
            "-i", temp_output_path,           # Input file from OpenCV
            "-c:v", "libx264",               # H.264 video codec
            "-preset", "fast",               # Faster encoding
            "-c:a", "aac",                   # AAC audio codec
            "-movflags", "+faststart",       # Optimize for web streaming
            "-y",                            # Overwrite output file
            output_video_path                # Final output file
        ], check=True)

        # Clean up the temporary OpenCV output
        os.remove(temp_output_path)

        result = [
            {"frame_idx": idx, "cx": cx, "cy": cy, "speed": speed}
            for idx, cx, cy, speed in tracked_positions
        ]
        output_video_url = f"/detections/{output_video_filename}"
        return {
            "tracked_positions": result,
            "output_video_url": output_video_url,
            "lower_hsv": lower_hsv,
            "upper_hsv": upper_hsv
        }
    finally:
        # Clean up temporary input files
        os.remove(temp_video_path)
        os.remove(temp_image_path)

@app.post("/track_orb")
async def track_orb(video: UploadFile = File(...), template_image: UploadFile = File(...)):
    """
    Endpoint to track an object in a video using ORB-based detection and save the annotated video in 'detections' folder.
    """
    # Temporary files for input
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(await video.read())
        temp_video_path = temp_video.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_image:
        temp_image.write(await template_image.read())
        temp_image_path = temp_image.name

    # Generate a unique output filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_video_filename = f"orb_detection_{timestamp}.mp4"
    output_video_path = os.path.join(DETECTIONS_DIR, output_video_filename)
    temp_output_path = output_video_path + ".tmp.mp4"  # Temporary file for OpenCV output

    try:
        # Process with save=True
        tracked_positions = overlay_detections_and_save_orb(
            temp_video_path, temp_image_path, temp_output_path, save=True
        )

        # Convert the video to H.264/AAC MP4 using FFmpeg
        subprocess.run([
            "ffmpeg",
            "-i", temp_output_path,           # Input file from OpenCV
            "-c:v", "libx264",               # H.264 video codec
            "-preset", "fast",               # Faster encoding
            "-c:a", "aac",                   # AAC audio codec
            "-movflags", "+faststart",       # Optimize for web streaming
            "-y",                            # Overwrite output file
            output_video_path                # Final output file
        ], check=True)

        # Clean up the temporary OpenCV output
        os.remove(temp_output_path)

        result = [
            {"frame_idx": idx, "cx": cx, "cy": cy, "speed": speed}
            for idx, cx, cy, speed in tracked_positions
        ]
        output_video_url = f"/detections/{output_video_filename}"
        return {
            "tracked_positions": result,
            "output_video_url": output_video_url
        }
    finally:
        # Clean up temporary input files
        os.remove(temp_video_path)
        os.remove(temp_image_path)

@app.post("/detect_dense_optical_flow")
async def detect_dense_optical_flow(video: UploadFile = File(...), image: UploadFile = File(...)):
    """
    Endpoint to detect motion using dense optical flow (Farneback) and save the annotated video in 'detections' folder.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(await video.read())
        temp_video_path = temp_video.name

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_video_filename = f"dense_optical_flow_{timestamp}.mp4"
    output_video_path = os.path.join(DETECTIONS_DIR, output_video_filename)
    temp_output_path = output_video_path + ".tmp.mp4"

    try:
        overlay_dense_optical_flow_and_save(temp_video_path, temp_output_path, save=True)
        subprocess.run([
            "ffmpeg",
            "-i", temp_output_path,
            "-c:v", "libx264",
            "-preset", "fast",
            "-c:a", "aac",
            "-movflags", "+faststart",
            "-y",
            output_video_path
        ], check=True)
        os.remove(temp_output_path)
        output_video_url = f"/detections/{output_video_filename}"
        return {"output_video_url": output_video_url}
    finally:
        os.remove(temp_video_path)

@app.post("/detect_sparse_optical_flow")
async def detect_sparse_optical_flow(video: UploadFile = File(...), image: UploadFile = File(...)):
    """
    Endpoint to detect motion using sparse optical flow (Lucas-Kanade) and save the annotated video in 'detections' folder.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(await video.read())
        temp_video_path = temp_video.name

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_video_filename = f"sparse_optical_flow_{timestamp}.mp4"
    output_video_path = os.path.join(DETECTIONS_DIR, output_video_filename)
    temp_output_path = output_video_path + ".tmp.mp4"

    try:
        overlay_sparse_optical_flow_and_save(temp_video_path, temp_output_path, save=True)
        subprocess.run([
            "ffmpeg",
            "-i", temp_output_path,
            "-c:v", "libx264",
            "-preset", "fast",
            "-c:a", "aac",
            "-movflags", "+faststart",
            "-y",
            output_video_path
        ], check=True)
        os.remove(temp_output_path)
        output_video_url = f"/detections/{output_video_filename}"
        return {"output_video_url": output_video_url}
    finally:
        os.remove(temp_video_path)

@app.post("/detect_background_subtraction")
async def detect_background_subtraction(video: UploadFile = File(...), image: UploadFile = File(...)):
    """
    Endpoint to detect motion using background subtraction and save the annotated video in 'detections' folder.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(await video.read())
        temp_video_path = temp_video.name

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_video_filename = f"background_subtraction_{timestamp}.mp4"
    output_video_path = os.path.join(DETECTIONS_DIR, output_video_filename)
    temp_output_path = output_video_path + ".tmp.mp4"

    try:
        overlay_background_subtraction_and_save(temp_video_path, temp_output_path, save=True)
        subprocess.run([
            "ffmpeg",
            "-i", temp_output_path,
            "-c:v", "libx264",
            "-preset", "fast",
            "-c:a", "aac",
            "-movflags", "+faststart",
            "-y",
            output_video_path
        ], check=True)
        os.remove(temp_output_path)
        output_video_url = f"/detections/{output_video_filename}"
        return {"output_video_url": output_video_url}
    finally:
        os.remove(temp_video_path)

@app.post("/detect_frame_differencing")
async def detect_frame_differencing(video: UploadFile = File(...), image: UploadFile = File(...)):
    """
    Endpoint to detect motion using frame differencing and save the annotated video in 'detections' folder.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(await video.read())
        temp_video_path = temp_video.name

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_video_filename = f"frame_differencing_{timestamp}.mp4"
    output_video_path = os.path.join(DETECTIONS_DIR, output_video_filename)
    temp_output_path = output_video_path + ".tmp.mp4"

    try:
        overlay_frame_differencing_and_save(temp_video_path, temp_output_path, save=True)
        subprocess.run([
            "ffmpeg",
            "-i", temp_output_path,
            "-c:v", "libx264",
            "-preset", "fast",
            "-c:a", "aac",
            "-movflags", "+faststart",
            "-y",
            output_video_path
        ], check=True)
        os.remove(temp_output_path)
        output_video_url = f"/detections/{output_video_filename}"
        return {"output_video_url": output_video_url}
    finally:
        os.remove(temp_video_path)