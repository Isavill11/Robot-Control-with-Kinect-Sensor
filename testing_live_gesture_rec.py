import os

import numpy as np
import pandas as pd
import mediapipe as mp
import cv2
import joblib
from pykinect2024 import PyKinectRuntime
from keras._tf_keras.keras.models import load_model
from google.protobuf.json_format import MessageToDict


### trying to see whether gesture recognition works or not


gesture_model = load_model('closer_data_coords_model.keras')
scaler = joblib.load('closer_data_coords_scaler.pkl')
GESTURE_CLASSES = os.listdir('C:/Users/Administrator/PycharmProjects/Kinect_body_tracking_v2/closer_data_coords')
remove = ['']

# for item in remove:
#     GESTURE_CLASSES.remove(item)
# print(GESTURE_CLASSES)

kinect = PyKinectRuntime.PyKinectRuntime(PyKinectRuntime.FrameSourceTypes_Color)
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hand = mp_hands.Hands()

landmark_columns = [f"lm_{i}_{axis}" for i in range(21) for axis in ["x", "y", "z"]]

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
                label = MessageToDict(results.multi_handedness[i])['classification'][0]['label']

                if label == "Right":
                    if len(hand_landmarks.landmark) == 21:  # Ensure 21 landmarks are detected
                        new_row = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark]).flatten()
                        new_row = new_row.reshape(1, -1)  # Ensure correct shape
                        # new_input_scaled = scaler.transform(new_row)

                        y_pred = gesture_model.predict(new_row)
                        y_pred_class = np.argmax(y_pred, axis=1)[0]

                        # print(new_row)
                        mp_drawing.draw_landmarks(color_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        # new_input_scaled = scaler.transform(new_row)q
                        print(GESTURE_CLASSES[y_pred_class])


                    else:
                        print("Warning: Hand landmarks detected but not 21 points!")
        cv2.imshow('live joint tracking', color_image)

    except Exception as e:
        print(f"Error: {e}")

    # Quit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
kinect.close()
cv2.destroyAllWindows()
