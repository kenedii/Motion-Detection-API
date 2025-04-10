import streamlit as st
import requests
import io

# API base URL (assuming FastAPI is running locally on port 8888)
API_URL = "http://localhost:8888"

# Page title
st.title("Object Tracking and Motion Detection in Video")

# Tracking method selection
method = st.radio("Select Method", (
    "Color Tracking",
    "ORB Tracking",
    "Sparse Optical Flow (Lucas-Kanade)",
    "Dense Optical Flow (Farneback)",
    "Background Subtraction",
    "Frame Differencing"
))

# File uploaders
video_file = st.file_uploader("Upload Video", type=["mp4"])

# Conditionally show image uploader only for methods that require it
if method in ["Color Tracking", "ORB Tracking"]:
    image_file = st.file_uploader("Upload Object Image", type=["png", "jpg", "jpeg"])
else:
    image_file = None  # Set to None for methods that don't use an image

# Process button
if st.button("Process"):
    # Check conditions based on method
    if method in ["Color Tracking", "ORB Tracking"]:
        if not video_file or not image_file:
            st.warning("Please upload both a video and an image file for this method.")
            st.stop()
    else:
        if not video_file:
            st.warning("Please upload a video file.")
            st.stop()

    with st.spinner("Processing..."):
        # Determine the API endpoint based on the method
        if method == "Color Tracking":
            url = f"{API_URL}/track_color"
            image_key = "object_image"
        elif method == "ORB Tracking":
            url = f"{API_URL}/track_orb"
            image_key = "template_image"
        elif method == "Sparse Optical Flow (Lucas-Kanade)":
            url = f"{API_URL}/detect_sparse_optical_flow"
            image_key = "image"
        elif method == "Dense Optical Flow (Farneback)":
            url = f"{API_URL}/detect_dense_optical_flow"
            image_key = "image"
        elif method == "Background Subtraction":
            url = f"{API_URL}/detect_background_subtraction"
            image_key = "image"
        elif method == "Frame Differencing":
            url = f"{API_URL}/detect_frame_differencing"
            image_key = "image"

        # Read video file contents
        video_bytes = video_file.read()

        # Prepare files for the POST request
        files = {
            'video': (video_file.name, io.BytesIO(video_bytes), 'video/mp4'),
        }

        # Add image file only if required
        if method in ["Color Tracking", "ORB Tracking"]:
            image_bytes = image_file.read()
            image_ext = image_file.name.split('.')[-1].lower()
            if image_ext in ['jpg', 'jpeg']:
                image_content_type = 'image/jpeg'
            elif image_ext == 'png':
                image_content_type = 'image/png'
            else:
                image_content_type = 'application/octet-stream'
            files[image_key] = (image_file.name, io.BytesIO(image_bytes), image_content_type)
        else:
            # For methods not requiring an image, send a dummy empty file to satisfy API
            files[image_key] = ('dummy.png', io.BytesIO(b''), 'image/png')

        # Send the request to the API
        response = requests.post(url, files=files)

        # Handle the response
        if response.status_code == 200:
            result = response.json()
            output_video_url = API_URL + result["output_video_url"]  # Full URL, e.g., http://localhost:8888/detections/...
            st.video(output_video_url)  # Use the URL directly

            if "tracked_positions" in result:
                tracked_positions = result["tracked_positions"]
                st.write("**Tracked Positions:**")
                st.table(tracked_positions)
                if method == "Color Tracking":
                    lower_hsv = result["lower_hsv"]
                    upper_hsv = result["upper_hsv"]
                    st.write(f"**Lower HSV Bounds:** {lower_hsv}")
                    st.write(f"**Upper HSV Bounds:** {upper_hsv}")
            else:
                st.write("Motion detection visualized. Speed calculation not available for this method.")
        else:
            st.error(f"Error processing the video: Status code {response.status_code}")