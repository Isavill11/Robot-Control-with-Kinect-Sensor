import os
import time
import cv2
import numpy as np
from pykinect2024 import PyKinect2024, PyKinectRuntime


FRAME_RATE = 30
DATA_DIR = 'object_detection_data'  #Change this to make a new file for the data.
CLASSES = ['1', '2']
DATASET_SIZE = 40 #This is the data size.

kinect_runtime = PyKinectRuntime.PyKinectRuntime(PyKinectRuntime.FrameSourceTypes_Color | PyKinectRuntime.FrameSourceTypes_Depth)


def kinect_color_frame():
    color_frame = kinect_runtime.get_last_color_frame()
    color_image = np.frombuffer(color_frame, dtype=np.uint8).reshape((1080, 1920, 4))
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGRA2BGR)
    resized_color_image = cv2.resize(color_image, (640, 480))
    return resized_color_image

# Make a data file, if theres alr one dont make it
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Make a class file inside the data file. if theres alr one dont make it
for class_name in CLASSES:
    class_dir = os.path.join(DATA_DIR, class_name)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

#this is the class we are collecting rn
for class_idx, class_name in enumerate(CLASSES):
    print('Collecting data for class {}'.format(class_name))

#just in case of an early quit, it will start collecting from whatever image number was last.
    class_dir = os.path.join(DATA_DIR, class_name)
    existing_images = [img for img in os.listdir(class_dir) if img.endswith('.jpg')]
    start_index = len(existing_images)
    print(f'starting from image index {start_index}')

    while True:
    #get frame from the kinect
        if kinect_runtime.has_new_color_frame():
            kinect_frame = kinect_color_frame()

        #Display ready message
            cv2.putText(kinect_frame, 'Ready? Press "Q" to start.', (100, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.imshow('Frame', kinect_frame)

        # wait for Q input to start collecting frames.
            if cv2.waitKey(25) == ord('q'):
                break

    #start storing data
    counter = 0
    while counter < DATASET_SIZE:

        if cv2.waitKey(25) == ord('f'):
            break

        start_time = time.time()

        if kinect_runtime.has_new_color_frame():
            kinect_frame = kinect_color_frame()

            image_path = os.path.join(DATA_DIR, class_name, f'{counter}.jpg')
            cv2.imwrite(image_path, kinect_frame)
            print(f'Saved: {image_path}')
            counter += 1

            cv2.imshow("Frame", kinect_frame)

        time_passed = time.time() - start_time
        sleep_time = 1
        time.sleep(sleep_time)

        

cv2.destroyAllWindows()


#PRESS F TO QUIT CODE.

#TODO: collect data of both humans and robot arm in different positions.
#TODO: collect data of human right next to the robot, with arm extended, walking by, standing far, etc.
#TODO: collect data of robotic arm in motion as well.
#TODO: Use labelstudio to annotate the data into boxes.
