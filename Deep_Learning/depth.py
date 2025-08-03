import ctypes
import sys
import time
import numpy as np
import cv2
from pykinect2024 import PyKinect2024
from pykinect2024 import PyKinectRuntime
import os

kinect_runtime = PyKinectRuntime.PyKinectRuntime(
    PyKinectRuntime.FrameSourceTypes_Depth | PyKinectRuntime.FrameSourceTypes_Color)


def kinect_depth_frame():
    depth_frame = kinect_runtime.get_last_depth_frame()  # Get the depth frame
    print(f"Depth frame raw data: {depth_frame[:10]}")  # Show first few values for debugging

    # Convert the depth frame into a numpy array
    depth_image = np.frombuffer(depth_frame, dtype=np.uint16)  # 16-bit depth data
    depth_image = depth_image.reshape((424, 512))  # Kinect One depth resolution (424x512)

    # Resize the depth frame to match the color frame (640x480)
    depth_image = cv2.resize(depth_image, (640, 480), interpolation=cv2.INTER_NEAREST)

    # Optionally scale depth values to make them more visible (for display purposes)
    depth_image = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
    depth_image = np.uint8(depth_image)  # Convert back to uint8 for display

    return depth_image


def kinect_color_frame():
    color_frame = kinect_runtime.get_last_color_frame()
    print(f"Color frame raw data: {color_frame[:10]}")  # Show first few values for debugging

    color_image = np.frombuffer(color_frame, dtype=np.uint8).reshape((1080, 1920, 4))
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGRA2BGR)
    resized_color_image = cv2.resize(color_image, (640, 480), interpolation=cv2.INTER_LINEAR)

    return resized_color_image


while True:
    if kinect_runtime.has_new_depth_frame() and kinect_runtime.has_new_color_frame():  # Check if there's a new depth and color frame
        depth_frame = kinect_depth_frame()  # Get the depth frame
        color_frame = kinect_color_frame()  # Get the color frame

        # Display the color and depth frames
        print(f"Depth frame shape: {depth_frame.shape}, Color frame shape: {color_frame.shape}")  # Debug shapes
        cv2.imshow('Color Frame', color_frame)
        cv2.imshow('Depth Frame', depth_frame)

        if cv2.waitKey(25) == ord('q'):  # Press 'q' to exit
            break

cv2.destroyAllWindows()  # Cleanup and close the window

while True:
    if kinect_runtime.has_new_depth_frame() and kinect_runtime.has_new_color_frame():  # Check if there's a new depth and color frame
        depth_frame = kinect_depth_frame()  # Get the depth frame
        color_frame = kinect_color_frame()  # Get the color frame

        # Display both frames side by side
        cv2.imshow('Color Frame', color_frame)
        cv2.imshow('Depth Frame', depth_frame)

        if cv2.waitKey(25) == ord('q'):  # Press 'q' to exit
            break

cv2.destroyAllWindows()  # Cleanup and close the window
