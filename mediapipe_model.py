import mediapipe as mp
import cv2
import os
import json

working_dir = os.getcwd()
# Paths
MAIN_PATH = working_dir+'/hand_data_extended'
OUTPUT_PATH = 'hand_data_extended_from_mediapipe'
GESTURE_CLASSES = os.listdir(MAIN_PATH)
FAILED_DATA = 'failed_data222'

# MediaPipe setup
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Ensure directories exist
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(FAILED_DATA, exist_ok=True)

# Process images
for GESTURE_CLASS in GESTURE_CLASSES:
   class_folder = os.path.join(MAIN_PATH, GESTURE_CLASS)
   if not os.path.exists(class_folder):
       print(f'Warning: folder {class_folder} does not exist.')
       continue

   image_files = [f for f in os.listdir(class_folder) if f.endswith('.jpg')]

   with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
       for idx, file_name in enumerate(image_files):
           image_path = os.path.join(class_folder, file_name)
           color_image = cv2.imread(image_path)

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

           if results.multi_hand_landmarks:
               hand_landmarks = results.multi_hand_landmarks[0]
               landmarks_dict = {}

               for idd, landmark in enumerate(hand_landmarks.landmark):
                   landmark_name = mp_hands.HandLandmark(idd).name
                   landmarks_dict[landmark_name] = {
                       "x": landmark.x,
                       "y": landmark.y
                   }

               # Create output class folder
               output_class_folder = os.path.join(OUTPUT_PATH, GESTURE_CLASS)
               os.makedirs(output_class_folder, exist_ok=True)

               # Save to JSON
               output_file = os.path.join(output_class_folder, f'landmarks_{file_name.replace(".jpg", ".json")}')
               with open(output_file, 'w') as f:
                   json.dump({
                       "gesture_class": GESTURE_CLASS,
                       "landmarks": landmarks_dict
                   }, f, indent=4)
               print(f"Saved landmarks for {file_name}")

           if not results.multi_hand_landmarks:
               print(f"Failed to detect hand landmarks in {file_name}")
               failed_folder = os.path.join(FAILED_DATA, GESTURE_CLASS)
               os.makedirs(failed_folder, exist_ok=True)
               failed_file = os.path.join(failed_folder, f'failed_{idx:03d}.json')

               with open(failed_file, 'w') as f:
                   json.dump({'gesture_class': GESTURE_CLASS, "image_path": image_path}, f, indent=4)

               print(f"Logged failed image in {failed_file}")




