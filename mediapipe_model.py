import mediapipe as mp
import numpy as np
import pandas as pd
import json
import cv2
import os


DATA_DIR = 'C:/Users/Administrator/PycharmProjects/Kinect_body_tracking_v2/NEW_hand_data'
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic


images_path = 'C:/Users/Administrator/PycharmProjects/Kinect_body_tracking_v2/hand_data/B-stop'
IMAGE_FILES = [f for f in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, f))]
for idx, file in enumerate(IMAGE_FILES):
        print(idx, file)
        image = cv2.imread(os.path.join(images_path, file))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with mp_holistic.Holistic(static_image_mode=True,
                                  model_complexity=2,
                                  enable_segmentation=True) as holistic:

                results = holistic.process(image)

                img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                mp_drawing.draw_landmarks(img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

                # cv2.imwrite('C:/Users/Administrator/PycharmProjects/Kinect_body_tracking_v2/annotated_image' + str(idx) + '.png', img)

                if results:
                        cv2.imwrite('C:/Users/Administrator/PycharmProjects/Kinect_body_tracking_v2/annotated_image' + str(idx) + '.png', img)