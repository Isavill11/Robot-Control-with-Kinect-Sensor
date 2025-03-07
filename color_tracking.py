import mediapipe as mp
import cv2
import os
import numpy as np
import json
from pykinect2024 import PyKinect2024, PyKinectRuntime

# MediaPipe setup
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

kinect = PyKinectRuntime.PyKinectRuntime(
    PyKinectRuntime.FrameSourceTypes_Color | PyKinectRuntime.FrameSourceTypes_Depth)

def get_latest_frame():
    # if there is a frame to collect, then collect it.
    if kinect.has_new_color_frame():
        # get latest frame
        color_frame = kinect.get_last_color_frame()
        # adjust color from bgra to bgr then resize image
        color_image = np.frombuffer(color_frame, dtype=np.uint8).reshape((1080, 1920, 4))
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGRA2BGR)
        new_frame = cv2.resize(color_image, (640, 480))
        return new_frame

# Define colors
LANDMARK_COLOR = (0, 255, 0)  # Green
CONNECTION_COLOR = (255, 0, 0)  # Blue



# Function to visualize landmarks on an image
def draw_landmarks(image, landmarks, image_width, image_height):
   for joint, coords in landmarks.items():
       x, y = int(coords["x"] * image_width), int(coords["y"] * image_height)
       cv2.circle(image, (x, y), 5, LANDMARK_COLOR, -1)  # Draw joint


   # Draw connections between keypoints
   hand_connections = mp_hands.HAND_CONNECTIONS
   for connection in hand_connections:
       joint1, joint2 = connection
       joint1_name = mp_hands.HandLandmark(joint1).name  # Convert index to name
       joint2_name = mp_hands.HandLandmark(joint2).name  # Convert index to name
       if joint1_name in landmarks and joint2_name in landmarks:
           x1, y1 = int(landmarks[joint1_name]["x"] * image_width), int(landmarks[joint1_name]["y"] * image_height)
           x2, y2 = int(landmarks[joint2_name]["x"] * image_width), int(landmarks[joint2_name]["y"] * image_height)
           cv2.line(image, (x1, y1), (x2, y2), CONNECTION_COLOR, 2)

while True:
    try:
        color_image = get_latest_frame()
        image_height, image_width, _ = color_image.shape

        with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
                rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_image)

    except AttributeError as e:
        print(f"Error: {e}")