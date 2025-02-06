from ultralytics import YOLO
import cv2
import numpy as np
from pykinect2024 import PyKinect2024, PyKinectRuntime
import time
import subprocess

CLASSES = {0: ["Person", (204,0,0)], #red
           1: ["Robot", (0, 128, 255)],} #blue

COLOR_TO_TRACK = {"lower_range": np.array([35, 80, 150]),
                  "upper_range": np.array([85, 255, 255])} # the hsv range for light green

model = YOLO("C:/Users/Administrator/PycharmProjects/Kinect_body_tracking_v2/runs/detect/train/weights/best.pt")

# Initialize Kinect Runtime
kinect_runtime = PyKinectRuntime.PyKinectRuntime(
    PyKinectRuntime.FrameSourceTypes_Color | PyKinectRuntime.FrameSourceTypes_Depth)

def get_latest_frame():
    # if there is a frame to collect, then collect it.
    if kinect_runtime.has_new_color_frame() and kinect_runtime.has_new_depth_frame():
        # get latest frame
        color_frame = kinect_runtime.get_last_color_frame()
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


wait = input("Enter anything to start and fyi press 'q' to quit")
frame_rate = 30  # Set target frame rate (frames per second)
frame_interval = 1 / frame_rate

while True:
    newest_frame = get_latest_frame()
    results = model.predict(newest_frame, verbose = False)

    for result in results: #just gives us access to all the data.
        for box in result.boxes:
            ### every value of box in the result.boxes attribute is an object prediction.
            ### so to access every object we iterate through the .boxes attribute. then we access all info from it such as
            ### confidence, class, bounding box coords, etc.
            x1, y1, x2, y2 = map(int, box.xyxy[0]) #bounding boxes coordinates
            # if x1 < 0 or y1 < 0 or x2 > newest_frame.shape[1] or y2 > newest_frame.shape[0]:
            #     print("Invalid bounding box coordinates!")
            #     continue

            confidence = box.conf[0]
            obj_class = int(box.cls[0])
            label = f"{CLASSES[obj_class][0]} {confidence:.2f}" #class label and confidence level

            ##display bounding box for corresponding class on the imaginary screen.
            cv2.rectangle(newest_frame, (x1, y1), (x2, y2), CLASSES[obj_class][1])
            cv2.putText(newest_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=CLASSES[obj_class][1], thickness=2)

            if check_for_color(newest_frame, x1, y1, x2, y2, COLOR_TO_TRACK["lower_range"], COLOR_TO_TRACK["upper_range"]):
                if CLASSES[obj_class][0] == "Person":
                    print(f"color detected in bounding box:{x1},{y1},{x2},{y2}.")
                    cv2.putText(newest_frame, "Authorized User!!", (x1+10, y1+50), cv2.FONT_HERSHEY_PLAIN, 2, color = (0, 255, 0), thickness=4)

    cv2.imshow("Object Tracking", newest_frame) ## this updates the screen so we can see the imaginary screen now.


    if cv2.waitKey(33) & 0xFF == ord('q'):
        break

kinect_runtime.close()
cv2.destroyAllWindows()






