from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response, StreamingResponse
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
from face_detection import (
    haar_cascade_face_detector,
    dlib_facial_analysis
)
import numpy as np
import cv2
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with Streamlit URL after deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

@app.post("/detect_faces_haar_video")
async def detect_faces_haar_video(video: UploadFile = File(...)):
    """Process a video with Haar Cascade face detection, stream it, and delete files."""
    # Save uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(await video.read())
        temp_video_path = temp_video.name

    # Generate output filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_video_filename = f"haar_faces_{timestamp}.mp4"
    output_video_path = os.path.join(DETECTIONS_DIR, output_video_filename)
    temp_output_path = output_video_path + ".tmp.mp4"  # Temporary file for OpenCV

    try:
        # Open the video
        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            raise Exception("Could not open video")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create VideoWriter for output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

        # Process each frame
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            processed_frame = haar_cascade_face_detector(frame)
            out.write(processed_frame)

        # Release resources
        cap.release()
        out.release()

        # Convert to H.264/AAC MP4 using FFmpeg
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

        # Read the output video into memory
        with open(output_video_path, "rb") as video_file:
            video_content = video_file.read()

        # Create a StreamingResponse to send the video
        response = StreamingResponse(
            content=iter([video_content]),
            media_type="video/mp4",
            headers={
                "Content-Disposition": f"attachment; filename={output_video_filename}"
            }
        )

        # Clean up files
        os.remove(temp_output_path)
        os.remove(output_video_path)
        os.remove(temp_video_path)

        return response
    except Exception as e:
        # Ensure cleanup on error
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)
        if os.path.exists(output_video_path):
            os.remove(output_video_path)
        raise e

@app.post("/detect_faces_dlib_video")
async def detect_faces_dlib_video(video: UploadFile = File(...)):
    """Process a video with Dlib facial analysis, stream it, and delete files."""
    # Save uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(await video.read())
        temp_video_path = temp_video.name

    # Generate output filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_video_filename = f"dlib_faces_{timestamp}.mp4"
    output_video_path = os.path.join(DETECTIONS_DIR, output_video_filename)
    temp_output_path = output_video_path + ".tmp.mp4"  # Temporary file for OpenCV

    try:
        # Open the video
        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            raise Exception("Could not open video")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create VideoWriter for output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))

        # Process each frame
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            processed_frame = dlib_facial_analysis(frame)
            out.write(processed_frame)

        # Release resources
        cap.release()
        out.release()

        # Convert to H.264/AAC MP4 using FFmpeg
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

        # Read the output video into memory
        with open(output_video_path, "rb") as video_file:
            video_content = video_file.read()

        # Create a StreamingResponse to send the video
        response = StreamingResponse(
            content=iter([video_content]),
            media_type="video/mp4",
            headers={
                "Content-Disposition": f"attachment; filename={output_video_filename}"
            }
        )

        # Clean up files
        os.remove(temp_output_path)
        os.remove(output_video_path)
        os.remove(temp_video_path)

        return response
    except Exception as e:
        # Ensure cleanup on error
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)
        if os.path.exists(output_video_path):
            os.remove(output_video_path)
        raise e