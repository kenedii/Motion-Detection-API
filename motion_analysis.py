import cv2
import numpy as np

def compute_hsv_bounds(object_image_path, num_clusters=3, min_area_ratio=0.1):
    """
    Compute HSV bounds for the most prominent color (assumed to be the object) using K-means clustering.

    Parameters:
        object_image_path (str): Path to the object image.
        num_clusters (int): Number of color clusters to identify (default 3 to separate object from background).
        min_area_ratio (float): Minimum ratio of pixels to consider a cluster significant.

    Returns:
        tuple: (lower_hsv, upper_hsv) for the dominant color cluster, adjusted with margins.
    """
    object_image = cv2.imread(object_image_path)
    if object_image is None:
        raise FileNotFoundError(f"Object image not found: {object_image_path}")

    # Convert to HSV
    hsv = cv2.cvtColor(object_image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Reshape for K-means (flatten to 1D per channel)
    pixels = hsv.reshape(-1, 3).astype(np.float32)

    # Apply K-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert centers to uint8
    centers = centers.astype(np.uint8)

    # Count pixels per cluster
    label_counts = np.bincount(labels.flatten(), minlength=num_clusters)
    total_pixels = len(labels)

    # Find the dominant cluster (most pixels, but not too small)
    dominant_cluster = None
    max_count = 0
    for i in range(num_clusters):
        count = label_counts[i]
        if count > max_count and count / total_pixels >= min_area_ratio:
            max_count = count
            dominant_cluster = i

    if dominant_cluster is None:
        # Fallback to manual orange range if clustering fails
        return np.array([5, 130, 80], dtype=np.uint8), np.array([15, 255, 255], dtype=np.uint8)

    # Get the HSV values of the dominant cluster
    h_center, s_center, v_center = centers[dominant_cluster]

    # Define margins to expand the range (tighter for Hue, broader for S and V)
    margins = (5, 30, 30)  # H, S, V
    h_min = max(0, h_center - margins[0])
    h_max = min(179, h_center + margins[0])
    s_min = max(0, s_center - margins[1])
    s_max = min(255, s_center + margins[1])
    v_min = max(0, v_center - margins[2])
    v_max = min(255, v_center + margins[2])

    lower_hsv = np.array([h_min, s_min, v_min], dtype=np.uint8)
    upper_hsv = np.array([h_max, s_max, v_max], dtype=np.uint8)
    return lower_hsv, upper_hsv

def detect_object_by_color(frame, lower_hsv, upper_hsv, area_threshold=50):
    """
    Detect the object in the frame using HSV thresholding.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > area_threshold:
            x, y, w, h = cv2.boundingRect(largest)
            M = cv2.moments(largest)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + w // 2, y + h // 2
            return (x, y, w, h), (cx, cy), mask
    return None, None, mask

def stabilize_frames(frames):
    """
    Stabilize video frames using feature tracking and affine transformation.
    """
    stabilized_frames = []
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    stabilized_frames.append(frames[0])

    for i in range(1, len(frames)):
        curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30)
        if prev_pts is None:
            m = np.eye(2, 3, dtype=np.float32)
        else:
            curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
            if curr_pts is None or len(curr_pts) < 1:
                m = np.eye(2, 3, dtype=np.float32)
            else:
                idx = np.where(status == 1)[0]
                prev_pts = prev_pts[idx]
                curr_pts = curr_pts[idx]
                m, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts)
                if m is None:
                    m = np.eye(2, 3, dtype=np.float32)

        frame_stabilized = cv2.warpAffine(frames[i], m, (frames[i].shape[1], frames[i].shape[0]))
        stabilized_frames.append(frame_stabilized)
        prev_gray = curr_gray

    return stabilized_frames

def overlay_detections_and_save_color(video_path, object_image_path,
                                      output_video_path=None, save=True):
    """
    Track the object in the video using color detection, overlay per‑frame speed,
    running average speed, timestamps, and (optionally) save the annotated video.

    Returns
    -------
    tracked_positions : list  (frame_idx, cx, cy, speed)
    lower_hsv, upper_hsv : list, list  – the HSV bounds actually used
    """
    lower_hsv, upper_hsv = compute_hsv_bounds(object_image_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video not found: {video_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ── read all frames ──────────────────────────────────────────────────────────
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    stabilized_frames = stabilize_frames(frames)

    # ── video writer (optional) ─────────────────────────────────────────────────
    out = None
    if save and output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    tracked_positions = []        # (frame_idx, cx, cy, speed)
    prev_centroid     = None
    speed_history     = []        # collect non‑zero speeds for running average

    for frame_idx, frame in enumerate(stabilized_frames):
        bbox, centroid, mask = detect_object_by_color(frame, lower_hsv, upper_hsv)
        frame_out = frame.copy()

        # ── timestamp overlay ───────────────────────────────────────────────────
        timestamp = frame_idx / fps
        cv2.putText(frame_out, f"Time: {timestamp:.2f}s", (10, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # ── object detected? ────────────────────────────────────────────────────
        if bbox is not None and centroid is not None:
            (x, y, w, h) = bbox
            (cx, cy)     = centroid

            # speed (pixel displacement between successive centroids)
            speed = 0.0
            if prev_centroid is not None:
                px, py = prev_centroid
                speed  = np.hypot(cx - px, cy - py)

            tracked_positions.append((frame_idx, cx, cy, speed))
            prev_centroid = centroid

            if speed > 0:
                speed_history.append(speed)

            # draw bbox / centroid / speed
            cv2.rectangle(frame_out, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.circle(frame_out, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(frame_out, f"Speed: {speed:.2f} px/frame",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame_out, "Object not detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        # ── running average speed overlay ───────────────────────────────────────
        avg_speed = np.mean(speed_history) if speed_history else 0.0
        cv2.putText(frame_out, f"Avg Speed: {avg_speed:.2f} px/frame",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 2, cv2.LINE_AA)

        # ── save / display ──────────────────────────────────────────────────────
        if out is not None:
            out.write(frame_out)

    if out is not None:
        out.release()

    return tracked_positions, lower_hsv.tolist(), upper_hsv.tolist()


def detect_object_by_orb(frame, template_path, orb, ratio_thresh=0.75, min_matches=10):
    """
    Detect the object in the frame using ORB feature matching.
    """
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if template is None:
        raise FileNotFoundError(f"Template image not found: {template_path}")
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_template, des_template = orb.detectAndCompute(template, None)
    kp_frame, des_frame = orb.detectAndCompute(frame_gray, None)

    if des_frame is None or des_template is None:
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des_template, des_frame, k=2)

    good_matches = [m for m, n in matches if m.distance < ratio_thresh * n.distance]
    if len(good_matches) < min_matches:
        return None

    pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches])
    centroid = np.mean(pts, axis=0)
    h, w = template.shape[:2]
    x = int(centroid[0] - w / 2)
    y = int(centroid[1] - h / 2)
    bbox = (x, y, w, h)
    score = len(good_matches)

    return {
        'bbox': bbox,
        'centroid': (int(centroid[0]), int(centroid[1])),
        'score': score
    }

def overlay_detections_and_save_orb(video_path, template_path, output_video_path=None, save=True):
    """
    Track the object in the video using ORB detection, overlay detections and timestamps, and optionally save.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video not found: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    stabilized_frames = stabilize_frames(frames)

    out = None
    if save and output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    orb = cv2.ORB_create(nfeatures=500)
    tracked_positions = []
    prev_centroid = None
    frame_idx = 0

    for frame in stabilized_frames:
        detection = detect_object_by_orb(frame, template_path, orb)
        frame_out = frame.copy()

        timestamp = frame_idx / fps
        timestamp_str = f"Time: {timestamp:.2f}s"
        cv2.putText(frame_out, timestamp_str, (10, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        if detection is not None:
            x, y, w, h = detection['bbox']
            cx, cy = detection['centroid']
            speed = 0.0
            if prev_centroid is not None:
                prev_cx, prev_cy = prev_centroid
                displacement = np.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
                speed = displacement
            tracked_positions.append((frame_idx, cx, cy, speed))
            prev_centroid = (cx, cy)

            cv2.rectangle(frame_out, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.circle(frame_out, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(frame_out, f"Speed: {speed:.2f} px/frame", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame_out, "Object not detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        if out is not None:
            out.write(frame_out)

        frame_idx += 1

    if out is not None:
        out.release()

    return tracked_positions

def overlay_dense_optical_flow_and_save(video_path, output_video_path=None, save=True):
    """
    Compute dense optical flow using Farneback's algorithm, visualize it with speed, and save the output video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video not found: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    stabilized_frames = stabilize_frames(frames)

    out = None
    if save and output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    prev_gray = None
    tracked_positions = []  # (frame_idx, x, y, speed)

    for frame_idx, frame in enumerate(stabilized_frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_out = frame.copy()

        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv = np.zeros((height, width, 3), dtype=np.uint8)
            hsv[..., 1] = 255
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            # Calculate average speed from magnitude
            avg_speed = np.mean(mag)
            tracked_positions.append((frame_idx, width//2, height//2, avg_speed))
            
            frame_out = cv2.addWeighted(frame, 0.5, flow_bgr, 0.5, 0)
            cv2.putText(frame_out, f"Avg Speed: {avg_speed:.2f} px/frame", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        timestamp = frame_idx / fps
        timestamp_str = f"Time: {timestamp:.2f}s"
        cv2.putText(frame_out, timestamp_str, (10, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        if out is not None:
            out.write(frame_out)

        prev_gray = gray

    if out is not None:
        out.release()
    
    return tracked_positions

def overlay_sparse_optical_flow_and_save(video_path, output_video_path=None, save=True):
    """
    Compute sparse optical flow using Lucas-Kanade algorithm, visualize tracked points with speed, and save the output video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video not found: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    stabilized_frames = stabilize_frames(frames)

    out = None
    if save and output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    prev_gray = cv2.cvtColor(stabilized_frames[0], cv2.COLOR_BGR2GRAY)
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
    mask = np.zeros_like(stabilized_frames[0])
    tracked_positions = []  # (frame_idx, x, y, speed)

    for frame_idx, frame in enumerate(stabilized_frames):
        frame_out = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_pts is not None:
            curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **lk_params)
            if curr_pts is not None:
                good_new = curr_pts[status == 1]
                good_old = prev_pts[status == 1]

                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    speed = np.sqrt((a - c)**2 + (b - d)**2)
                    tracked_positions.append((frame_idx, a, b, speed))
                    mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
                    frame_out = cv2.circle(frame_out, (int(a), int(b)), 5, (0, 0, 255), -1)
                    cv2.putText(frame_out, f"{speed:.2f}", (int(a), int(b) - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

                frame_out = cv2.add(frame_out, mask)
                prev_pts = good_new.reshape(-1, 1, 2)
            else:
                prev_pts = None
        else:
            prev_pts = cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)

        timestamp = frame_idx / fps
        timestamp_str = f"Time: {timestamp:.2f}s"
        cv2.putText(frame_out, timestamp_str, (10, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        if out is not None:
            out.write(frame_out)

        prev_gray = gray

    if out is not None:
        out.release()
    
    return tracked_positions

def overlay_background_subtraction_and_save(video_path, output_video_path=None, save=True):
    """
    Perform background subtraction using the first frame as background, detect motion with speed, and save the output video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video not found: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    stabilized_frames = stabilize_frames(frames)

    background = stabilized_frames[0]
    background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    background_gray = cv2.GaussianBlur(background_gray, (21, 21), 0)

    out = None
    if save and output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    tracked_positions = []  # (frame_idx, cx, cy, speed)
    prev_centroids = {}

    for frame_idx, frame in enumerate(stabilized_frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        diff = cv2.absdiff(background_gray, gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.dilate(thresh, kernel, iterations=2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        frame_out = frame.copy()
        curr_centroids = {}
        
        for i, contour in enumerate(contours):
            if cv2.contourArea(contour) > 500:
                x, y, w, h = cv2.boundingRect(contour)
                cx = x + w // 2
                cy = y + h // 2
                speed = 0.0
                
                if frame_idx > 0 and prev_centroids:
                    min_dist = float('inf')
                    for prev_id, (px, py) in prev_centroids.items():
                        dist = np.sqrt((cx - px)**2 + (cy - py)**2)
                        if dist < min_dist:
                            min_dist = dist
                            speed = dist
                curr_centroids[i] = (cx, cy)
                tracked_positions.append((frame_idx, cx, cy, speed))
                
                cv2.rectangle(frame_out, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame_out, f"Speed: {speed:.2f}", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        timestamp = frame_idx / fps
        timestamp_str = f"Time: {timestamp:.2f}s"
        cv2.putText(frame_out, timestamp_str, (10, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        
        if out is not None:
            out.write(frame_out)
        
        prev_centroids = curr_centroids

    if out is not None:
        out.release()
    
    return tracked_positions

def overlay_frame_differencing_and_save(video_path, output_video_path=None, save=True):
    """
    Perform frame differencing to detect motion with speed and save the output video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video not found: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    stabilized_frames = stabilize_frames(frames)

    out = None
    if save and output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    prev_gray = None
    tracked_positions = []  # (frame_idx, cx, cy, speed)
    prev_centroids = {}

    for frame_idx, frame in enumerate(stabilized_frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        frame_out = frame.copy()
        
        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            thresh = cv2.dilate(thresh, kernel, iterations=2)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            curr_centroids = {}
            for i, contour in enumerate(contours):
                if cv2.contourArea(contour) > 500:
                    x, y, w, h = cv2.boundingRect(contour)
                    cx = x + w // 2
                    cy = y + h // 2
                    speed = 0.0
                    
                    if prev_centroids:
                        min_dist = float('inf')
                        for prev_id, (px, py) in prev_centroids.items():
                            dist = np.sqrt((cx - px)**2 + (cy - py)**2)
                            if dist < min_dist:
                                min_dist = dist
                                speed = dist
                    curr_centroids[i] = (cx, cy)
                    tracked_positions.append((frame_idx, cx, cy, speed))
                    
                    cv2.rectangle(frame_out, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame_out, f"Speed: {speed:.2f}", (x, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            
            prev_centroids = curr_centroids

        timestamp = frame_idx / fps
        timestamp_str = f"Time: {timestamp:.2f}s"
        cv2.putText(frame_out, timestamp_str, (10, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        
        if out is not None:
            out.write(frame_out)
        
        prev_gray = gray

    if out is not None:
        out.release()
    
    return tracked_positions

# Backward compatibility wrappers
def track_object_in_video(video_path, object_image_path, output_video_path=None):
    tracked_positions, _, _ = overlay_detections_and_save_color(video_path, object_image_path, output_video_path, save=False)
    return tracked_positions

