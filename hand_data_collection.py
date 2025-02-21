import cv2
import os
import numpy as np
from ultralytics import YOLO
from pykinect2024 import PyKinectRuntime, PyKinect2024
import time

# Constants
working_dir = os.getcwd()
FRAME_RATE = 1
DATASET_SIZE = 100  # Per class
BASE_DIR = 'idk'
HAND_CLASSES = ['A-sign', 'B-sign', 'C-sign', 'peace_sign', 'thumbs_up', 'thumbs_down', 'palm_up', 'Okay']
OBJ_CLASSES = {0: ["Person", (204,0,0)], #red
               1: ["Robot", (0, 128, 255)], #blue
               }
# Initialize YOLO model and Kinect
model = YOLO("C:/Users/Administrator/PycharmProjects/Kinect_body_tracking_v2/runs/detect/train/weights/best.pt")
kinect = PyKinectRuntime.PyKinectRuntime(PyKinect2024.FrameSourceTypes_Color)

def kinect_color_frame():
    color_frame = kinect.get_last_color_frame()
    color_image = np.frombuffer(color_frame, dtype=np.uint8).reshape((1080, 1920, 4))
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGRA2BGR)
    resized_color_image = cv2.resize(color_image, (640, 480))
    return resized_color_image

def kinect_depth_frame():
    depth_frame = kinect.get_last_depth_frame()
    depth_image = np.frombuffer(depth_frame, dtype=np.uint16).reshape((1080, 1920))
    depth_image = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return depth_image

def process_and_save(frame, x1, y1, x2, y2, save_path, index):
    """Crop the frame to the AOI, resize, and save."""
    if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
        print("Invalid bounding box coordinates!")
        return index
    aoi = frame[y1:y2, x1:x2]  # Crop AOI
    # zoomed_aoi = cv2.resize(aoi, (224, 224))  # Resize for consistency
    file_path = os.path.join(save_path, f'image_{index:04d}.jpg')
    cv2.imwrite(file_path, aoi)
    return index + 1

# Ensure directories exist
if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)
for class_name in HAND_CLASSES:
    class_dir = os.path.join(BASE_DIR, class_name)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)


for class_name in HAND_CLASSES:
    print(f'Collecting data for class {class_name}')
    class_dir = os.path.join(BASE_DIR, class_name)
    existing_images = [img for img in os.listdir(class_dir) if img.endswith('.jpg')] ### check if already images in the file.
    start_index = len(existing_images)

    collecting = False
    print(f"Starting from image index {start_index}")

    while True:
        if kinect.has_new_color_frame():
            frame = kinect_color_frame()
            results = model.predict(frame, verbose=False)

            ### this block of code will allow us to display the object prediction regardless
            ### of whether we are collecting images or not, for user convenience.
            for result in results:
                for box in result.boxes:
                    confidence = box.conf[0]
                    obj_class = int(box.cls[0])

                    ### only get the bounding boxes of a person object and if the confidence is high.
                    if confidence >= 0.65 and OBJ_CLASSES[obj_class][0] == "Person":
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        label = f"{OBJ_CLASSES[obj_class][0]} {confidence:.2f}"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), OBJ_CLASSES[obj_class][1], 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, OBJ_CLASSES[obj_class][1], 2)

                        if not collecting:
                            cv2.putText(frame, 'Press Q to start collecting...', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        if collecting and start_index < DATASET_SIZE:
                            start_index = process_and_save(frame, x1, y1, x2, y2, class_dir, start_index)
                            time.sleep(0.1)
                            print( f'Saved image {start_index} for class {class_name} with bounding box coordinates: {x1},{y1},{x2},{y2}')

            cv2.imshow('Data Collection', frame)

            # checks if collecting or not collecting.
            key = cv2.waitKey(25)
            if key == ord('q') and not collecting:
                time.sleep(2)
                collecting = True
            elif key == ord('q') and collecting:
                collecting = False
                break
            elif key == ord('f'):
                quit()


        if start_index >= DATASET_SIZE:
            print(f"Completed collecting {DATASET_SIZE} images for {class_name}.")
            break

kinect.close()
cv2.destroyAllWindows()




