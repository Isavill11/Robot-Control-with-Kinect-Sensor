import cv2
import os
import numpy as np
from ultralytics import YOLO
from pykinect2024 import PyKinectRuntime, PyKinect2024
import time


## constants
FRAME_RATE = 1
DATASET_SIZE = 20  # Per class
BASE_DIR = 'AA_experiment'
IMG_TYPE = ['color', 'depth']
HAND_CLASSES = ['A-start', 'B-stop', 'C-pos_1']
OBJ_CLASSES = {0: ["Person", (204,0,0)], #red
              1: ["Robot", (0, 128, 255)], #blue
              }

# init yolo model and kinect now
model = YOLO("C:/Users/Administrator/PycharmProjects/Kinect_body_tracking_v2/runs/detect/train/weights/best.pt")
kinect = PyKinectRuntime.PyKinectRuntime(PyKinect2024.FrameSourceTypes_Color | PyKinect2024.FrameSourceTypes_Depth)


def kinect_color_frame():
   color_frame = kinect.get_last_color_frame()
   color_image = np.frombuffer(color_frame, dtype=np.uint8).reshape((1080, 1920, 4))
   color_image = cv2.cvtColor(color_image, cv2.COLOR_BGRA2BGR)
   resized_color_image = cv2.resize(color_image, (640, 480))
   return resized_color_image

def kinect_depth_frame():
    depth_frame = kinect.get_last_depth_frame()  # Get the depth frame

    # Convert the depth frame into a numpy array
    depth_image = np.frombuffer(depth_frame, dtype=np.uint16).reshape((424, 512))  # 16-bit depth data then Kinect One depth resolution (424x512)
    resized_depth = cv2.resize(depth_image, (640, 480))

    # Optionally scale depth values to make them more visible (for display purposes)
    resized_depth = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
    resized_depth = np.uint8(depth_image)  # Convert back to uint8 for display

    return resized_depth


def process_and_save(color_frame, depth_frame, x1, y1, x2, y2, color_save_path, depth_save_path, index):
   """Crop the frame to the AOI, resize, and save."""
   ##check if bounding boxes are valid
   if x1 < 0 or y1 < 0 or x2 > color_frame.shape[1] or y2 > color_frame.shape[0]:
       print("Invalid bounding box coordinates!")
       return index
   aoi_color = color_frame[y1:y2, x1:x2]
   aoi_depth = depth_frame[y1:y2, x1:x2]

   color_file_path = os.path.join(color_save_path, f'color_image_{index:04d}.jpg')
   depth_file_path = os.path.join(depth_save_path, f'depth_image_{index:04d}.png')

   cv2.imwrite(color_file_path, aoi_color)
   cv2.imwrite(depth_file_path, aoi_depth)
   return index + 1

# check if directories exist
if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)
    # create each img type folder in base directory, then create each class folder inside the img type folder
    for classes in IMG_TYPE:
       img_type = os.path.join(classes)
       # print(f'Creating directories for {img_type} images...')
       for class_name in HAND_CLASSES:
           class_dir = os.path.join(BASE_DIR, img_type, class_name)
           if not os.path.exists(class_dir):
               os.makedirs(class_dir)
               print(f'Created directory for {class_name} images.')


## starting main loop
for class_name in HAND_CLASSES:
   # print(f'Collecting data for class {class_name}')
   class_dir_color = os.path.join(BASE_DIR, IMG_TYPE[0], class_name)
   class_dir_depth = os.path.join(BASE_DIR, IMG_TYPE[1], class_name)
   existing_images = [img for img in os.listdir(class_dir_color) if img.endswith('.jpg')] ### check if already images in the file.
   start_index = len(existing_images)

   collecting = False
   print(f"Starting from image index {start_index}")
   print("remember, press 'f' to quit the entire program.")

   while True:
       if kinect.has_new_color_frame() and kinect.has_new_depth_frame():
           color_frame = kinect_color_frame()
           depth_frame = kinect_depth_frame()
           results = model.predict(color_frame, verbose=False) ## predict using color image

           ### this block of code will allow us to display the object prediction regardless
           ### of whether we are collecting images or not, for user convenience.
           for result in results:
               for box in result.boxes:
                   confidence = box.conf[0]
                   obj_class = int(box.cls[0])

                   ### only get the bounding boxes of a person object if confident to reduce lag.
                   if confidence >= 0.45 and OBJ_CLASSES[obj_class][0] == "Person":
                       x1, y1, x2, y2 = map(int, box.xyxy[0])
                       label = f"{OBJ_CLASSES[obj_class][0]} {confidence:.2f}"
                       cv2.rectangle(color_frame, (x1, y1), (x2, y2), OBJ_CLASSES[obj_class][1], 2)
                       cv2.putText(color_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, OBJ_CLASSES[obj_class][1], 2)

                       ### check if we are collecting or not here so we can always display the predicted result and reduce redundancy.
                       if not collecting:
                           cv2.putText(color_frame, 'Press Q to start collecting...', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                       if collecting and start_index < DATASET_SIZE:
                           start_index = process_and_save(color_frame, depth_frame, x1, y1, x2, y2, class_dir_color, class_dir_depth, start_index)
                           time.sleep(1)
                           print( f'Saved image {start_index} for class {class_name} with bounding box coordinates: {x1},{y1},{x2},{y2}')


           ##after everything, display the image.
           cv2.imshow('Data Collection', color_frame)

           # listens to see if we are collecting or not collecting. This is inside the while loop.
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
           break ## this brings us back to the for loop, changing the class we are collecting data for.

## cleanup
kinect.close()
cv2.destroyAllWindows()




