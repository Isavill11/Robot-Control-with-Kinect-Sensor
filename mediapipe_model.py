import mediapipe as mp
import cv2
import os
import json


# Paths
MAIN_PATH = 'C:/Users/Administrator/PycharmProjects/Kinect_body_tracking_v2/hand_data'
# DEPTH_IMAGES_PATH = 'path_to_depth_images'
OUTPUT_PATH = 'data_for_gesture_rec'
GESTURE_CLASSES = ["A-start", "B-stop", "C-pos_1"]
FAILED_DATA = 'failed_data'

# MediaPipe setup
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


if not os.path.exists(OUTPUT_PATH):
   os.makedirs(OUTPUT_PATH)
if not os.path.exists(FAILED_DATA):
   os.makedirs(FAILED_DATA)


# Process images
for GESTURE_CLASS in GESTURE_CLASSES:
   class_folder = os.path.join(MAIN_PATH, GESTURE_CLASS)
   if not os.path.exists(class_folder):
       print(f'warning: folder {class_folder} does not exist.')
       continue


   image_files = [f for f in os.listdir(class_folder) if f.endswith('.jpg')]


   with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
       for idx, file_name in enumerate(image_files):
           image_path = os.path.join(class_folder, file_name)
           color_image = cv2.imread(image_path)
           # depth_image = cv2.imread(os.path.join(DEPTH_IMAGES_PATH, file_name.replace('.jpg', '.png')),
           #                          cv2.IMREAD_UNCHANGED)


           if color_image is None:
               print(f"Error reading image {file_name}")
               failed_folder = os.path.join(FAILED_DATA, GESTURE_CLASS, f'failed_{idx:03d}.json')
               if not os.path.exists(os.path.dirname(failed_folder)):
                   os.makedirs(os.path.dirname(failed_folder))
                   with open(failed_folder, 'w') as f:
                       json.dump({'gesture_class': GESTURE_CLASS, "image_path": image_path}, f)
                   continue




           image_height, image_width, _ = color_image.shape


           # Convert to RGB for MediaPipe
           rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)


           # Process with MediaPipe
           results = hands.process(rgb_image)


           ### IF there are results, add and save them to the output folder we created.
           ### ELSE if there are NO results, add that image path to another folder dedicated to the failed image.
           if results.multi_hand_landmarks:
               # Extract landmarks
               hand_landmarks = results.multi_hand_landmarks[0]
               landmark_data = []


               for landmark in hand_landmarks.landmark:
                   pixel_x = int(landmark.x * image_width)
                   pixel_y = int(landmark.y * image_height)
                   landmark_data.append({
                       "x": landmark.x,
                       "y": landmark.y,
                       # "z": depth_value
                   })

               # create output class folder
               output_class_folder = os.path.join(OUTPUT_PATH, GESTURE_CLASS)
               if not os.path.exists(output_class_folder):
                   os.makedirs(output_class_folder)

               #save to json.
               output_file = os.path.join(OUTPUT_PATH, GESTURE_CLASS, f'landmarks_{idx:03d}.json')
               with open(output_file, 'w') as f:
                   json.dump({'gesture_class': GESTURE_CLASS,
                              "landmarks": landmark_data,}, f)
               print(f"Saved landmarks for image {file_name}")


           else:
               failed_folder = os.path.join(FAILED_DATA, GESTURE_CLASS, f'failed_{idx:03d}.json')

               if not os.path.exists(os.path.dirname(failed_folder)):
                   os.makedirs(os.path.dirname(failed_folder))

               file_location = os.path.join(class_folder, file_name)
               with open(failed_folder, 'w') as f:
                   json.dump({'gesture_class': GESTURE_CLASS, "image_path": file_location}, f)
               print(f"Failed to detect hand landmarks in {file_name}. Logged in {failed_folder}.")


