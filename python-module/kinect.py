from pykinect2024 import PyKinect2024, PyKinectRuntime
from obj_ges_models import *
from body import *
import cv2
import mediapipe
import numpy as np
import pandas as pd



class MyKinect(PyKinectRuntime.PyKinectRuntime): 
    CAMERA_TYPES = {
        'Color': PyKinectRuntime.FrameSourceTypes_Color, 
        'Infrared': PyKinectRuntime.FrameSourceTypes_Infrared,
        'Depth': PyKinectRuntime.FrameSourceTypes_Depth, 
        'Body': PyKinectRuntime.FrameSourceTypes_Body                
    }
    
    def __init__(self, camera_type='Color'):
        if isinstance(camera_type, str): 
            camera_type = self.CAMERA_TYPES[camera_type]
        super().__init__(camera_type)
        self.camera_type = camera_type

    def change_camera(self, new_camera_type): 
        if isinstance(new_camera_type, str): 
            new_camera_type = self.CAMERA_TYPES[new_camera_type]
        super().__init__(new_camera_type)
        self.camera_type = new_camera_type

    def get_latest_frame(self):
        # if there is a frame to collect, then collect it.
        if self.has_new_color_frame() and self.has_new_depth_frame():
            # get latest frame
            color_frame = self.get_last_color_frame()
            # adjust color from bgra to bgr then resize image
            color_image = np.frombuffer(color_frame, dtype=np.uint8).reshape((1080, 1920, 4))
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGRA2BGR)
            new_frame = cv2.resize(color_image, (640, 480))
            return new_frame
        return None

    @staticmethod
    def check_for_color(roi, lower_hsv_range, upper_hsv_range, threshold):
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, lower_hsv_range, upper_hsv_range)
        color_count = cv2.countNonZero(mask)
        return color_count > threshold

    @staticmethod
    def kinect_color_frame(self):
        color_frame = self.get_last_color_frame()
        color_image = np.frombuffer(color_frame, dtype=np.uint8).reshape((1080, 1920, 4))
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGRA2BGR)
        resized_color_image = cv2.resize(color_image, (640, 480))
        return resized_color_image
    
    @staticmethod
    def process_and_save(frame, x1, y1, x2, y2, save_path, index):
        if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
            print("Invalid bounding box coordinates!")
            return index
        aoi = frame[y1:y2, x1:x2]  # Crop AOI
        # zoomed_aoi = cv2.resize(aoi, (224, 224))  # Resize for consistency
        file_path = os.path.join(save_path, f'image_{index:04d}.jpg')
        cv2.imwrite(file_path, aoi)
        return index + 1