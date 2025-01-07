
import cv2
import numpy as np
from pykinect2024 import PyKinect2024, PyKinectRuntime
import time
import subprocess


# Initialize Kinect Runtime
kinect_runtime = PyKinectRuntime.PyKinectRuntime(PyKinectRuntime.FrameSourceTypes_Color | PyKinectRuntime.FrameSourceTypes_Depth)

wait = input("Enter anything to start ")

frame_rate = 30  # Set target frame rate (frames per second)

while True:
    #if there is a frame to collect, then collect it.
    if kinect_runtime.has_new_color_frame() and kinect_runtime.has_new_depth_frame():
        #get latest frame
        color_frame = kinect_runtime.get_last_color_frame()
        # depth_frame = kinect_runtime.get_last_depth_frame()
        # process color from bgra to bgr then to hsv
        #adjust frame from bgra to bgr then resize
        color_image = np.frombuffer(color_frame, dtype=np.uint8).reshape((1080, 1920, 4))
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGRA2BGR)
        resized_color_image = cv2.resize(color_image, (640, 480))

        #track person.

        #Track color
        hsv_color_frame = cv2.cvtColor(resized_color_image, cv2.COLOR_BGR2HSV)
        # track the color.
        color_lower = np.array([10, 150, 100], np.uint8)
        color_upper = np.array([50, 255, 255], np.uint8)
        color_mask = cv2.inRange(hsv_color_frame, color_lower, color_upper)

        kernel = np.ones((5, 5), np.uint8)
        color_mask = cv2.dilate(color_mask, kernel)
        res_green = cv2.bitwise_and(resized_color_image, resized_color_image, mask=color_mask)

        # find contours of the area
        contours, hierarchy = cv2.findContours(color_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Draw bounding boxes around detected green areas
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 300:  # Ignore small areas
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(resized_color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the processed image
        cv2.imshow('Kinect Color Frame', resized_color_image)

        # fps is 30.
        frame_time = time.time() - start_time
        wait_time = max(1, int(1000 / frame_rate - frame_time * 1000))

        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break

#end code
kinect_runtime.close()
cv2.destroyAllWindows()

