from ultralytics import YOLO
import cv2
import numpy as np
from pykinect2024 import PyKinect2024, PyKinectRuntime
import time
import subprocess

CLASSES = {0: ["Person", (204,0,0)],
           1: ["Robot", (0, 128, 255)]
           }
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

wait = input("Enter anything to start and fyi press 'q' to quit")
frame_rate = 30  # Set target frame rate (frames per second)
frame_interval = 1 / frame_rate

while True:

    newest_frame = get_latest_frame()
    results = model.predict(newest_frame)


    for result in results: #just gives us access to all the data.
        for box in result.boxes:
            ### every value of box in the result.boxes attribute is an object prediction.
            ### so to access every object we iterate through the .boxes attribute. then we access all info from it such as
            ### confidence, class, bouding box coords, etc.
            x1, y1, x2, y2 = map(int, box.xyxy[0]) #bounding boxes coordinates
            confidence = box.conf[0]
            obj_class = int(box.cls[0])
            label = f"{CLASSES[obj_class][0]} {confidence:.2f}" #class label and confidence level

            ##display bounding box for corresponding class on the imaginary screen.
            cv2.rectangle(newest_frame, (x1, y1), (x2, y2), CLASSES[obj_class][1])
            cv2.putText(newest_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CLASSES[obj_class][1], 2)

    cv2.imshow("Object Tracking", newest_frame) ## this updates the screen so we can see the imaginary screen now.

    if cv2.waitKey(int(frame_interval *1000)) & 0xFF == ord('q'):
        break

kinect_runtime.close()
cv2.destroyAllWindows()






