import mediapipe as mp
import cv2
import os
import json


# Paths
COLOR_IMAGES_PATH = 'path_to_color_images'
DEPTH_IMAGES_PATH = 'path_to_depth_images'
OUTPUT_PATH = 'path_to_output_json'
GESTURE_CLASS = "B-stop"


# MediaPipe setup
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


# Process images
image_files = [f for f in os.listdir(COLOR_IMAGES_PATH) if f.endswith('.jpg')]


if not os.path.exists(OUTPUT_PATH):
   os.makedirs(OUTPUT_PATH)


with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
   for idx, file_name in enumerate(image_files):
       # Load images
       color_image = cv2.imread(os.path.join(COLOR_IMAGES_PATH, file_name))
       depth_image = cv2.imread(os.path.join(DEPTH_IMAGES_PATH, file_name.replace('.jpg', '.png')),
                                cv2.IMREAD_UNCHANGED)


       image_height, image_width, _ = color_image.shape


       # Convert to RGB for MediaPipe
       rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)


       # Process with MediaPipe
       results = hands.process(rgb_image)


       if results.multi_hand_landmarks:
           # Extract landmarks
           hand_landmarks = results.multi_hand_landmarks[0]
           landmark_data = []


           for landmark in hand_landmarks.landmark:
               pixel_x = int(landmark.x * image_width)
               pixel_y = int(landmark.y * image_height)


               # Handle out-of-bounds cases
               if pixel_x < 0 or pixel_y < 0 or pixel_x >= depth_image.shape[1] or pixel_y >= depth_image.shape[0]:
                   depth_value = 0  # Assign default depth value for invalid coordinates
               else:
                   depth_value = depth_image[pixel_y, pixel_x]


               landmark_data.append({
                   "x": landmark.x,
                   "y": landmark.y,
                   "z": depth_value
               })


           # Save to JSON
           output_file = os.path.join(OUTPUT_PATH, f'landmarks_{idx:04d}.json')
           with open(output_file, 'w') as f:
               json.dump({'gesture_class': GESTURE_CLASS,
                          "landmarks": landmark_data,}, f)


           print(f"Saved landmarks for image {file_name}")
