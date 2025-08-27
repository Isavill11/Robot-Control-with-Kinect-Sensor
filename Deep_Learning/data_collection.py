import os
import time
import sys
import cv2
import numpy as np
from pykinect2024 import PyKinect2024, PyKinectRuntime
from ultralytics import YOLO
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from kinect import MyKinect
# collects both object and hand gesture data


OD_FRAME_RATE = 30
OD_DATA_DIR = 'object_detection_data'  #Change this to make a new file for the data.
OD_CLASSES = ['1', '2']
OD_DATASET_SIZE = 40 #This is the data size.

WORKING_DIR = os.getcwd()
GR_FRAME_RATE= 1
DATASET_SIZE = 100  # Per class
GR_DATA_DIR = 'closer_data'
HAND_CLASSES = ['A-sign', 'B-sign', 'peace_sign', 'thumbs_up', 'Okay']

OBJ_CLASSES = {0: ["Person", (204,0,0)], #red
               1: ["Robot", (0, 128, 255)], #blue
               }
model = YOLO("C:/Users/Administrator/PycharmProjects/Kinect_body_tracking_v2/runs/detect/train/weights/best.pt")
kinect = MyKinect()



option = input("what kind of data are you collecting? Gesture data or object data? (choose g or o)")

if option.lower() == 'o':
    if not os.path.exists(OD_DATA_DIR):
        os.makedirs(OD_DATA_DIR)

    for class_name in OD_CLASSES:
        class_dir = os.path.join(OD_DATA_DIR, class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
    #this is the class we are collecting rn
    for class_idx, class_name in enumerate(OD_CLASSES):
        print('Collecting data for class {}'.format(class_name))

    #just in case of an early quit, it will start collecting from whatever image number was last.
        class_dir = os.path.join(OD_DATA_DIR, class_name)
        existing_images = [img for img in os.listdir(class_dir) if img.endswith('.jpg')]
        start_index = len(existing_images)
        print(f'starting from image index {start_index}')

        while True:
            if kinect.has_new_color_frame():
                kinect_frame = kinect.kinect_color_frame()

                cv2.putText(kinect_frame, 'Ready? Press "Q" to start.', (100, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
                cv2.imshow('Frame', kinect_frame)
                if cv2.waitKey(25) == ord('q'):
                    break
            counter = 0

        while counter < OD_DATASET_SIZE:

            if cv2.waitKey(25) == ord('f'):
                quit()

            start_time = time.time()
            if kinect.has_new_color_frame():
                kinect_frame = kinect.kinect_color_frame()

                image_path = os.path.join(OD_DATA_DIR, class_name, f'{counter}.jpg')
                cv2.imwrite(image_path, kinect_frame)
                print(f'Saved: {image_path}')
                counter += 1

                cv2.imshow("Frame", kinect_frame)

            time_passed = time.time() - start_time
            sleep_time = 1
            time.sleep(sleep_time)
            ### PRESS F TO QUIT CODE

elif option.lower() == 'g':
    if not os.path.exists(GR_DATA_DIR):
        os.makedirs(GR_DATA_DIR)
    for class_name in HAND_CLASSES:
        class_dir = os.path.join(GR_DATA_DIR, class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

    for class_name in HAND_CLASSES:
        print(f'Collecting data for class {class_name}')
        class_dir = os.path.join(GR_DATA_DIR, class_name)
        existing_images = [img for img in os.listdir(class_dir) if img.endswith('.jpg')] ### check if already images in the file.
        start_index = len(existing_images)

        collecting = False
        print(f"Starting from image index {start_index}")

        while True:
            if kinect.has_new_color_frame():
                frame = kinect.kinect_color_frame()
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
                                start_index = kinect.process_and_save(frame, x1, y1, x2, y2, class_dir, start_index)
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
