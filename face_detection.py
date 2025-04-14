import dlib
import cv2
import numpy as np
import os
import bz2
import urllib.request

def haar_cascade_face_detector(image): # scan the image at multiple scales and apply a pre-trained model 
# Load the pre-trained Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,  # Adjusts how much the image size is reduced at each image scale
        minNeighbors=5,   # Defines how many neighbors each candidate rectangle should have to retain it
        minSize=(30, 30)  # Minimum possible object size. Objects smaller than this are ignored.
    )

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the resulting image with detected faces
    return image


def dlib_facial_analysis(image):
# Advanced facial analysis using DLib’s pre-trained 68-landmark predictor and face detector


    # Create a directory for models if it doesn't exist
    os.makedirs('models', exist_ok=True)

    # Define the URL and file paths
    model_url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    compressed_model_path = "models/shape_predictor_68_face_landmarks.dat.bz2"
    model_path = "models/shape_predictor_68_face_landmarks.dat"

    # Download the compressed model if it doesn't exist
    if not os.path.exists(model_path):
        print("Downloading facial landmark predictor model...")
        urllib.request.urlretrieve(model_url, compressed_model_path)
        
        # Extract the compressed file
        print("Extracting compressed model file...")
        with bz2.BZ2File(compressed_model_path) as input_file, open(model_path, 'wb') as output_file:
            output_file.write(input_file.read())
        
        print(f"Model saved to {model_path}")

      # Helper function to calculate Eye Aspect Ratio (EAR) for blink detection
    def eye_aspect_ratio(eye):
      A = np.linalg.norm(eye[1] - eye[5])  # Distance between vertical landmarks
      B = np.linalg.norm(eye[2] - eye[4])
      C = np.linalg.norm(eye[0] - eye[3])  # Distance between horizontal landmarks
      ear = (A + B) / (2.0 * C)
      return ear

    # Load the predictor and detector
    predictor = dlib.shape_predictor(model_path)
    detector = dlib.get_frontal_face_detector()

    # Convert image to grayscale for detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)

    # Convert dlib's rectangle to OpenCV style
    def rect_to_bb(rect):
        x = rect.left()
        y = rect.top()
        w = rect.right() - x
        h = rect.bottom() - y
        return (x, y, w, h)

    # Convert landmark prediction to numpy array
    def shape_to_np(shape):
        landmarks = np.zeros((68, 2), dtype=int)
        for i in range(0, 68):
            landmarks[i] = (shape.part(i).x, shape.part(i).y)
        return landmarks

    # Process each face
    for rect in rects:
        # Get face rectangle
        (x, y, w, h) = rect_to_bb(rect)
        
        # Draw face rectangle
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Get landmarks
        shape = predictor(gray, rect)
        landmarks = shape_to_np(shape)

        # Draw landmarks
        for (lx, ly) in landmarks:
            cv2.circle(image, (lx, ly), 2, (0, 0, 255), -1)

        # Face alignment: Calculate angle using eye centers
        left_eye_points = landmarks[36:42]
        right_eye_points = landmarks[42:48]
        left_eye_center = np.mean(left_eye_points, axis=0).astype(int)
        right_eye_center = np.mean(right_eye_points, axis=0).astype(int)
        cv2.line(image, tuple(left_eye_center), tuple(right_eye_center), (255, 0, 0), 2)
        dy = right_eye_center[1] - left_eye_center[1]
        dx = right_eye_center[0] - left_eye_center[0]
        angle = np.degrees(np.arctan2(dy, dx))

        # Blink detection: Calculate EAR for both eyes
        left_ear = eye_aspect_ratio(left_eye_points)
        right_ear = eye_aspect_ratio(right_eye_points)
        avg_ear = (left_ear + right_ear) / 2.0
        blink_text = "Blinking" if avg_ear < 0.2 else "Eyes Open"

        # Expression recognition: Check mouth width
        mouth_left = landmarks[48]
        mouth_right = landmarks[54]
        mouth_width = np.linalg.norm(mouth_left - mouth_right)
        expression_text = "Smiling" if mouth_width > w * 0.3 else "Neutral"

        # Display analysis results below the face
        text_y = y + h + 20
        cv2.putText(image, f"Rotation: {angle:.1f} degrees", (x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        text_y += 20
        cv2.putText(image, blink_text, (x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        text_y += 20
        cv2.putText(image, expression_text, (x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Return the image with analysis displayed
    return image