
import os
import cv2
import joblib
import time
import subprocess
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
from pykinect2024 import PyKinect2024, PyKinectRuntime
from keras._tf_keras.keras.models import load_model
from google.protobuf.json_format import MessageToDict
from control_scorbot import execute_matlab_commands


CLASSES = {0: ["Person", (204,0,0)], #red
           1: ["Robot", (0, 128, 255)],} #blue

COLOR_TO_TRACK = {"lower_range": np.array([10, 150, 150]),
                  "upper_range": np.array([25, 255, 255])} # the rgb range for whatever color
threshold = 40  ##min number of pixels needed to determine if the colors present.

LANDMARK_COLOR = (0, 255, 0)  # Green
CONNECTION_COLOR = (255, 0, 0)  # Blue
GESTURE_CLASSES = os.listdir('C:/Users/Administrator/PycharmProjects/Kinect_body_tracking_v2/closer_data_coords')

gesture_timer = None
current_gesture = None
GESTURE_HOLD_TIME = 3
last_trigger_time = 0
cooldown_duration = 10



obj_model = YOLO("C:/Users/Administrator/PycharmProjects/Kinect_body_tracking_v2/runs/detect/train/weights/best.pt")
gesture_model = load_model('_model2.keras')
scaler = joblib.load('_scaler2.pkl')
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hand = mp_hands.Hands()


kinect = PyKinectRuntime.PyKinectRuntime(
    PyKinectRuntime.FrameSourceTypes_Color | PyKinectRuntime.FrameSourceTypes_Depth)

def get_latest_frame():
    # if there is a frame to collect, then collect it.
    if kinect.has_new_color_frame() and kinect.has_new_depth_frame():
        # get latest frame
        color_frame = kinect.get_last_color_frame()
        # adjust color from bgra to bgr then resize image
        color_image = np.frombuffer(color_frame, dtype=np.uint8).reshape((1080, 1920, 4))
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGRA2BGR)
        new_frame = cv2.resize(color_image, (640, 480))
        return new_frame

def check_for_color(roi, lower_hsv_range, upper_hsv_range):
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, lower_hsv_range, upper_hsv_range)
    color_count = cv2.countNonZero(mask)

    return color_count > threshold

def joint_tracking(frame, roi):
    global gesture_timer, current_gesture
    predicted_gesture = None

    try:
        roi_width, roi_height, _ = roi.shape
        rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        results_gesture = hand.process(rgb_roi)

        if results_gesture.multi_hand_landmarks and results_gesture.multi_handedness:
            roi_mask = np.zeros_like(roi)

            for i, hand_landmarks in enumerate(results_gesture.multi_hand_landmarks):
                label = MessageToDict(results_gesture.multi_handedness[i])['classification'][0]['label']
                if label == "Right":
                    new_row = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark], dtype=np.float32).flatten().reshape(1, -1)
                    new_input_scaled = scaler.transform(new_row)

                    y_pred = gesture_model.predict(new_input_scaled)
                    y_pred_class = np.argmax(y_pred, axis=1)[0]
                    predicted_gesture = GESTURE_CLASSES[y_pred_class]  # Store predicted gesture

                for idx, landmark in enumerate(hand_landmarks.landmark):
                    x, y = int(landmark.x * roi_width), int(landmark.y * roi_height)
                    # cv2.circle(roi_mask, (x, y), 5, (0, 0, 255), -1)

                    if idx == 0 and predicted_gesture:
                        text_x, text_y = x - 20, y - 20  # text position slightly above wrist
                        cv2.putText(frame, predicted_gesture, (text_x, text_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                #draw landmarks onto the mask then overlay the mask onto the og frame.
                mp_drawing.draw_landmarks(roi_mask, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            frame[y1:y2, x1:x2] = cv2.addWeighted(frame[y1:y2, x1:x2], 1, roi_mask, 1, 0)

        return predicted_gesture

    except Exception as e:
        print(f"Error tracking joints: {e}")

while True:
    try:
        newest_frame = get_latest_frame()
        image_width, image_height, _  = newest_frame.shape
        results_obj = obj_model.predict(newest_frame, verbose = False)

        for result in results_obj: #just gives us access to all the data.
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0]) #bounding boxes coordinates
                if x1 < 0 or y1 < 0 or x2 > newest_frame.shape[1] or y2 > newest_frame.shape[0]:
                    print("Invalid bounding box coordinates!")
                else:
                    roi = newest_frame[y1:y2, x1:x2]

                confidence, obj_class = box.conf[0], int(box.cls[0])
                label = f"{CLASSES[obj_class][0]} {confidence:.2f}" #class label and confidence level

                if confidence > 0.40:
                    cv2.rectangle(newest_frame, (x1, y1), (x2, y2), CLASSES[obj_class][1])
                    cv2.putText(newest_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=CLASSES[obj_class][1], thickness=2)

                    if check_for_color(roi, COLOR_TO_TRACK["lower_range"], COLOR_TO_TRACK["upper_range"]) and CLASSES[obj_class][0] == "Person":
                        gesture = joint_tracking(newest_frame, roi)

                        if gesture:
                            if gesture == current_gesture:
                                if gesture_timer and time.time() - gesture_timer >= GESTURE_HOLD_TIME:
                                    if time.time() - last_trigger_time >= cooldown_duration:
                                        print(f"Gesture '{gesture}' held for 3 seconds. Triggering robotic arm.")
                                        execute_matlab_commands()
                                        last_trigger_time = time.time()
                                    else:
                                        print("Cooldown active. Gesture ignored.")
                                    gesture_timer = None
                                    current_gesture = None
                            else:
                                current_gesture = gesture
                                gesture_timer = time.time()
                        else:
                            current_gesture = None
                            gesture_timer = None
                        # print(f"color detected in bounding box:{x1},{y1},{x2},{y2}.")
                        # cv2.putText(newest_frame, "Authorized User!!", (x1+10, y1+50), cv2.FONT_HERSHEY_PLAIN, 2, color = (0, 255, 0), thickness=4)

        cv2.imshow("Object Tracking", newest_frame) ## this updates the screen so we can see the imaginary screen now.

    except AttributeError as e:
        print(f"Error: {e}. Cant capture frame right now.")
        continue

    if cv2.waitKey(33) & 0xFF == ord('q'):
        break

kinect.close()
cv2.destroyAllWindows()






