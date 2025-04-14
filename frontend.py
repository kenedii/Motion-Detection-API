import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image
import tempfile
import plotly.express as px
from streamlit_plotly_events import plotly_events
import requests
import io

# Set page layout to wide for better display
st.set_page_config(layout="wide")

# App title
st.title("Video Analysis App")

# API base URL (assuming FastAPI is running locally on port 8888)
API_URL = "http://localhost:8888"

# Task selection
task = st.selectbox("Select Task", ["Face detection", "Object motion detection", "Motion detection"])

# --- Face Detection ---
if task == "Face detection":
    face_method = st.radio("Select Face Detection Method", ["dlib", "haar cascade"])
    uploaded_video = st.file_uploader("Upload Video for Face Detection", type=["mp4", "avi", "mov"])
    if uploaded_video:
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
            tfile.write(uploaded_video.read())
            video_path = tfile.name
        
        # Open video capture
        cap = cv2.VideoCapture(video_path)
        try:
            if not cap.isOpened():
                st.error("Error: Could not open video.")
            else:
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_number = st.slider("Select Frame for Face Detection", 0, total_frames - 1, 0)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    st.image(frame_rgb, caption="Selected Frame", use_container_width=True)
                    if st.button("Detect Faces"):
                        # Convert frame to bytes for API
                        _, encoded_frame = cv2.imencode('.jpg', frame)
                        frame_bytes = encoded_frame.tobytes()
                        url = f"{API_URL}/dlib_facial_analysis" if face_method == "dlib" else f"{API_URL}/haar_cascade_face_detector"
                        files = {'file': ('frame.jpg', frame_bytes, 'image/jpeg')}
                        with st.spinner("Detecting faces..."):
                            response = requests.post(url, files=files)
                        if response.status_code == 200:
                            result_img = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
                            st.image(result_img, caption="Face Detection Result", use_container_width=True)
                        else:
                            st.error(f"Error in face detection: Status code {response.status_code}")
                else:
                    st.error("Error: Could not read frame.")
        finally:
            cap.release()
        # Clean up
        try:
            os.remove(video_path)
        except PermissionError as e:
            st.warning(f"Could not delete temporary file: {e}")

# --- Motion Detection ---
elif task == "Motion detection":
    motion_method = st.radio("Select Motion Detection Algorithm", [
        "Sparse Optical Flow (Lucas-Kanade)",
        "Dense Optical Flow (Farneback)",
        "Background Subtraction",
        "Frame Differencing"
    ])
    uploaded_video = st.file_uploader("Upload Video for Motion Detection", type=["mp4", "avi", "mov"])
    if uploaded_video:
        st.write("**Video uploaded. Select an algorithm above and click 'Process' to start motion detection.**")
        if st.button("Process"):
            video_bytes = uploaded_video.read()
            # Map method to API endpoint
            method_to_url = {
                "Sparse Optical Flow (Lucas-Kanade)": "detect_sparse_optical_flow",
                "Dense Optical Flow (Farneback)": "detect_dense_optical_flow",
                "Background Subtraction": "detect_background_subtraction",
                "Frame Differencing": "detect_frame_differencing"
            }
            url = f"{API_URL}/{method_to_url[motion_method]}"
            files = {
                'video': (uploaded_video.name, io.BytesIO(video_bytes), 'video/mp4'),
                'image': ('dummy.png', io.BytesIO(b''), 'image/png')  # Dummy image as per API requirement
            }
            with st.spinner("Processing motion detection..."):
                response = requests.post(url, files=files)
            if response.status_code == 200:
                result = response.json()
                output_video_url = API_URL + result["output_video_url"]
                st.video(output_video_url)
                st.success("Motion detection completed.")
            else:
                st.error(f"Error processing the video: Status code {response.status_code}")

# --- Object Motion Detection ---
elif task == "Object motion detection":
    tracking_method = st.radio("Select Tracking Algorithm", ["Color Tracking", "ORB Tracking"])
    template_option = st.radio("Select Template Option", ["Upload Template Image", "Select from Video Frame"])

    if template_option == "Upload Template Image":
        uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
        uploaded_template = st.file_uploader("Upload Template Image", type=["png", "jpg", "jpeg"])
        if uploaded_video and uploaded_template and st.button("Process"):
            video_bytes = uploaded_video.read()
            template_bytes = uploaded_template.read()
            url = f"{API_URL}/track_color" if tracking_method == "Color Tracking" else f"{API_URL}/track_orb"
            image_key = "object_image" if tracking_method == "Color Tracking" else "template_image"
            files = {
                'video': (uploaded_video.name, io.BytesIO(video_bytes), 'video/mp4'),
                image_key: (uploaded_template.name, io.BytesIO(template_bytes), 'image/png')
            }
            with st.spinner("Processing object tracking..."):
                response = requests.post(url, files=files)
            if response.status_code == 200:
                result = response.json()
                output_video_url = API_URL + result["output_video_url"]
                st.video(output_video_url)
                if "tracked_positions" in result:
                    st.write("**Tracked Positions:**", result["tracked_positions"])
                    if tracking_method == "Color Tracking":
                        st.write(f"**Lower HSV Bounds:** {result['lower_hsv']}")
                        st.write(f"**Upper HSV Bounds:** {result['upper_hsv']}")
                st.success("Object tracking completed.")
            else:
                st.error(f"Error processing the video: Status code {response.status_code}")

    else:  # Select from Video Frame
        uploaded_video = st.file_uploader("Upload Video for Template Selection", type=["mp4", "avi", "mov"])
        if uploaded_video:
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                tfile.write(uploaded_video.read())
                video_path = tfile.name

            # Initialize session state variables
            if 'selecting_template' not in st.session_state:
                st.session_state.selecting_template = False
            if 'template_selected' not in st.session_state:
                st.session_state.template_selected = False

            if not st.session_state.template_selected:
                if st.button("Select Template from Frame"):
                    st.session_state.selecting_template = True

                if st.session_state.selecting_template:
                    # Video capture
                    cap = cv2.VideoCapture(video_path)
                    try:
                        if not cap.isOpened():
                            st.error("Error: Could not open video.")
                        else:
                            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            frame_number = st.slider("Select Frame", 0, total_frames - 1, 0)
                            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                            ret, frame = cap.read()
                            if ret:
                                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                # Scale frame for display
                                max_display_width = 800
                                original_height, original_width, _ = frame_rgb.shape
                                scale_factor = max_display_width / original_width if original_width > max_display_width else 1.0
                                display_width = int(original_width * scale_factor)
                                display_height = int(original_height * scale_factor)
                                display_frame = cv2.resize(frame_rgb, (display_width, display_height))

                                # Interactive Plotly display
                                fig = px.imshow(display_frame)
                                fig.update_layout(
                                    margin=dict(l=0, r=0, t=0, b=0),
                                    height=display_height,
                                    width=display_width
                                )
                                selected_points = plotly_events(fig, click_event=True)

                                # Process clicked points
                                if selected_points:
                                    for point in selected_points:
                                        x = int(point['x'] / scale_factor)
                                        y = int(point['y'] / scale_factor)
                                        x = max(0, min(original_width - 1, x))
                                        y = max(0, min(original_height - 1, y))
                                        if 'clicked_points' not in st.session_state:
                                            st.session_state.clicked_points = []
                                        if len(st.session_state.clicked_points) < 2:
                                            st.session_state.clicked_points.append((x, y))

                                # Display points and bounding box
                                if 'clicked_points' in st.session_state and st.session_state.clicked_points:
                                    points = st.session_state.clicked_points
                                    x_points = [p[0] * scale_factor for p in points]
                                    y_points = [p[1] * scale_factor for p in points]
                                    fig.add_scatter(x=x_points, y=y_points, mode='markers', marker=dict(color='red', size=10))
                                    if len(points) == 2:
                                        p1, p2 = points
                                        x_min, x_max = sorted([p1[0], p2[0]])
                                        y_min, y_max = sorted([p1[1], p2[1]])
                                        st.session_state.crop_coords = (x_min, y_min, x_max, y_max)
                                        cropped_preview = frame_rgb[y_min:y_max, x_min:x_max]
                                        st.image(cropped_preview, caption="Cropped Template Preview", use_container_width=True)
                                        st.write("Review the cropped template above. Click below to confirm or cancel.")
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            if st.button("Finalize Template"):
                                                st.session_state.cropped_template = cropped_preview
                                                st.session_state.template_selected = True
                                                st.session_state.selecting_template = False
                                                if 'clicked_points' in st.session_state:
                                                    del st.session_state.clicked_points
                                                if 'crop_coords' in st.session_state:
                                                    del st.session_state.crop_coords
                                                st.success("Template finalized! You can now process the detection.")
                                        with col2:
                                            if st.button("Cancel Selection"):
                                                if 'clicked_points' in st.session_state:
                                                    del st.session_state.clicked_points
                                                if 'crop_coords' in st.session_state:
                                                    del st.session_state.crop_coords
                                                st.info("Selection cancelled. You can select a new region or choose another option.")
                            else:
                                st.error("Error: Could not read frame.")
                    finally:
                        cap.release()  # Ensure the VideoCapture is released

                    st.write("Click two points to define the template region.")

            else:
                # Template selected, show it and process option
                st.image(st.session_state.cropped_template, caption="Selected Template", use_container_width=True)
                st.write("**Template finalized. Click 'Process' to start detection.**")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Reselect Template"):
                        st.session_state.template_selected = False
                        st.session_state.selecting_template = False
                        if 'clicked_points' in st.session_state:
                            del st.session_state.clicked_points
                        if 'crop_coords' in st.session_state:
                            del st.session_state.crop_coords
                with col2:
                    if st.button("Process"):
                        # Convert cropped template to bytes
                        cropped_pil = Image.fromarray(st.session_state.cropped_template)
                        buf = io.BytesIO()
                        cropped_pil.save(buf, format="PNG")
                        template_bytes = buf.getvalue()
                        with open(video_path, "rb") as f:
                            video_bytes = f.read()
                        url = f"{API_URL}/track_color" if tracking_method == "Color Tracking" else f"{API_URL}/track_orb"
                        image_key = "object_image" if tracking_method == "Color Tracking" else "template_image"
                        files = {
                            'video': ('video.mp4', io.BytesIO(video_bytes), 'video/mp4'),
                            image_key: ('template.png', io.BytesIO(template_bytes), 'image/png')
                        }
                        with st.spinner("Processing object tracking..."):
                            response = requests.post(url, files=files)
                        if response.status_code == 200:
                            result = response.json()
                            output_video_url = API_URL + result["output_video_url"]
                            st.video(output_video_url)
                            if "tracked_positions" in result:
                                st.write("**Tracked Positions:**", result["tracked_positions"])
                                if tracking_method == "Color Tracking":
                                    st.write(f"**Lower HSV Bounds:** {result['lower_hsv']}")
                                    st.write(f"**Upper HSV Bounds:** {result['upper_hsv']}")
                            st.success("Object tracking completed.")
                        else:
                            st.error(f"Error processing the video: Status code {response.status_code}")
                        # Clean up temporary file
                        try:
                            os.remove(video_path)
                        except PermissionError as e:
                            st.warning(f"Could not delete temporary file: {e}")

else:
    st.info("Please select a task and upload the required files to start.")