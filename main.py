
from ultralytics import YOLO
from ultralytics import YOLOWorld
import cv2
import numpy as np
from pykinect2024 import PyKinect2024, PyKinectRuntime
from keras._tf_keras.keras.models import load_model
import mediapipe as mp
from google.protobuf.json_format import MessageToDict

import time
import subprocess



CLASSES = {0: ["Person", (204,0,0)], #red
           1: ["Robot", (0, 128, 255)],} #blue

COLOR_TO_TRACK = {"lower_range": np.array([10, 150, 150]),
                  "upper_range": np.array([25, 255, 255])} # the rgb range for whatever color
LANDMARK_COLOR = (0, 255, 0)  # Green
CONNECTION_COLOR = (255, 0, 0)  # Blue


obj_model = YOLO("C:/Users/Administrator/PycharmProjects/Kinect_body_tracking_v2/runs/detect/train/weights/best.pt")
gesture_model = load_model('gesture_model.h5')
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hand = mp_hands.Hands()


# Initialize Kinect Runtime
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

def check_for_color(frame, x1, y1, x2, y2, lower_hsv_range, upper_hsv_range):
    #only check the area of the objects bounding box.
    if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
        return
    roi = frame[y1:y2, x1:x2]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV) #turn into a Hue Saturation Value for easier color classification
    mask = cv2.inRange(hsv_roi, lower_hsv_range, upper_hsv_range) #the color range im searching for.

    color_count = cv2.countNonZero(mask)
    threshold = 40 ##min number of pixels needed to determine if the colors present.

    return color_count > threshold

def joint_tracking(frame, x1, y1, x2, y2):
    try:
        if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
            return

        roi = frame[y1:y2, x1:x2]
        roi_width, roi_height = x2 - x1, y2 - y1

        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        results = hand.process(roi_rgb)

        if results.multi_hand_landmarks:

            for i in results.multi_handedness:
                label = MessageToDict(i)['classification'][0]['label']
            for hand_landmarks in results.multi_hand_landmarks:
                # create a new blank mask for the ROI
                roi_mask = np.zeros_like(roi)

                for landmark in hand_landmarks.landmark:
                    # convert normalized landmark coordinates to ROI pixel coordinates
                    x = int(landmark.x * roi_width)
                    y = int(landmark.y * roi_height)

                    # Draw on the ROI mask
                    cv2.circle(roi_mask, (x, y), 5, (0, 0, 255), -1)

                mp_drawing.draw_landmarks(roi_mask, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                # Overlay the drawn landmarks onto the original frame
                frame[y1:y2, x1:x2] = cv2.addWeighted(frame[y1:y2, x1:x2], 1, roi_mask, 1, 0)

    except Exception as e:
        print(f"Error tracking joints: {e}")

while True:
    try:
        newest_frame = get_latest_frame()
        image_width, image_height, _  = newest_frame.shape

        results = obj_model.predict(newest_frame, verbose = False)

        for result in results: #just gives us access to all the data.
            for box in result.boxes:
                ### every value of box in the result.boxes attribute is an object prediction.
                ### so to access every object we iterate through the .boxes attribute. then we access all info from it such as
                ### confidence, class, bounding box coords, etc.
                x1, y1, x2, y2 = map(int, box.xyxy[0]) #bounding boxes coordinates
                if x1 < 0 or y1 < 0 or x2 > newest_frame.shape[1] or y2 > newest_frame.shape[0]:
                    print("Invalid bounding box coordinates!")
                    continue

                confidence = box.conf[0]
                obj_class = int(box.cls[0])
                label = f"{CLASSES[obj_class][0]} {confidence:.2f}" #class label and confidence level
                if confidence > 0.40:
                    ##display bounding box for corresponding class on the imaginary screen.
                    cv2.rectangle(newest_frame, (x1, y1), (x2, y2), CLASSES[obj_class][1])
                    cv2.putText(newest_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=CLASSES[obj_class][1], thickness=2)

                    if check_for_color(newest_frame, x1, y1, x2, y2, COLOR_TO_TRACK["lower_range"], COLOR_TO_TRACK["upper_range"]) and CLASSES[obj_class][0] == "Person":
                            ####
                            joint_tracking(newest_frame, x1, y1, x2, y2)
                            ###
                            print(f"color detected in bounding box:{x1},{y1},{x2},{y2}.")
                            cv2.putText(newest_frame, "Authorized User!!", (x1+10, y1+50), cv2.FONT_HERSHEY_PLAIN, 2, color = (0, 255, 0), thickness=4)

        cv2.imshow("Object Tracking", newest_frame) ## this updates the screen so we can see the imaginary screen now.

    except AttributeError as e:
        print(f"Error: {e}. Cant capture frame right now.")
        continue

    if cv2.waitKey(33) & 0xFF == ord('q'):
        break

kinect.close()
cv2.destroyAllWindows()






