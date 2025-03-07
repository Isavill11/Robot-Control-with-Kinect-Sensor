import numpy as np
import pandas as pd
import mediapipe as mp
import cv2
from pykinect2024 import PyKinectRuntime
from keras._tf_keras.keras.models import load_model
from google.protobuf.json_format import MessageToDict

# Load gesture recognition model
gesture_model = load_model('gesture_model2.h5')

# Kinect and Mediapipe setup
kinect = PyKinectRuntime.PyKinectRuntime(PyKinectRuntime.FrameSourceTypes_Color)
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hand = mp_hands.Hands()

# Define column names for the DataFrame (21 landmarks * 3 coordinates)
landmark_columns = [f"lm_{i}_{axis}" for i in range(21) for axis in ["x", "y", "z"]]
df = pd.DataFrame(columns=landmark_columns)  # Empty DataFrame to store gesture data

def get_latest_frame():
    """Retrieve the latest Kinect color frame."""
    if kinect.has_new_color_frame():
        color_frame = kinect.get_last_color_frame()
        color_image = np.frombuffer(color_frame, dtype=np.uint8).reshape((1080, 1920, 4))
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGRA2BGR)
        return cv2.resize(color_image, (640, 480))
    return None

while True:
    try:
        color_image = get_latest_frame()
        if color_image is None:
            continue

        image_height, image_width, _ = color_image.shape
        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        results = hand.process(rgb_image)

        if results.multi_hand_landmarks and results.multi_handedness:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Identify if this is the right hand
                label = MessageToDict(results.multi_handedness[i])['classification'][0]['label']
                if label != "Right":
                    continue  # Skip left hand

                # Extract landmark coordinates
                landmark_data = []
                for lm in hand_landmarks.landmark:
                    landmark_data.extend([lm.x, lm.y, lm.z])  # Flatten (x, y, z)

                # Convert to DataFrame format (1-row)
                new_row = pd.DataFrame([landmark_data], columns=landmark_columns)
                df = pd.concat([df, new_row], ignore_index=True)

                # Draw right-hand landmarks on the frame
                mp_drawing.draw_landmarks(color_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Pass DataFrame row to gesture model (if needed)
                prediction = gesture_model.predict(new_row)
                prediction_class = np.argmax(prediction, axis=1)
                print("Gesture Prediction:", prediction_class)  # Replace with actual processing

        # Display processed video feed
        cv2.imshow('live joint tracking', color_image)

    except Exception as e:
        print(f"Error: {e}")  # Print actual error for debugging

    # Quit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
kinect.close()
cv2.destroyAllWindows()
