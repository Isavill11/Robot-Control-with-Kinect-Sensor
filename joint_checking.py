
import json
import os
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np

## this uses mediapipe to draw and save the hand landmarks to a file, then allows you to see them before asking
## whether you want to save the file or not.

working_dir = os.getcwd()
MAIN_PATH = working_dir + '/closer_data'
print(MAIN_PATH)
JSON_PATH = working_dir + '/closer_data_coords'  # Path where JSON files with landmarks are stored
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
GESTURE_CLASSES = os.listdir(MAIN_PATH)
data_list = []


LANDMARK_COLOR = (0, 255, 0)  # Green
CONNECTION_COLOR = (255, 0, 0)  # Blue

def draw_landmarks(image, landmarks, image_width, image_height):
   for joint, coords in landmarks.items():
       x, y = int(coords["x"] * image_width), int(coords["y"] * image_height)
       cv2.circle(image, (x, y), 5, LANDMARK_COLOR, -1)  # Draw joint


   # Draw connections between keypoints
   hand_connections = mp_hands.HAND_CONNECTIONS
   for connection in hand_connections:
       joint1, joint2 = connection
       joint1_name = mp_hands.HandLandmark(joint1).name  # Convert index to name
       joint2_name = mp_hands.HandLandmark(joint2).name  # Convert index to name
       if joint1_name in landmarks and joint2_name in landmarks:
           x1, y1 = int(landmarks[joint1_name]["x"] * image_width), int(landmarks[joint1_name]["y"] * image_height)
           x2, y2 = int(landmarks[joint2_name]["x"] * image_width), int(landmarks[joint2_name]["y"] * image_height)
           cv2.line(image, (x1, y1), (x2, y2), CONNECTION_COLOR, 2)

def save_to_df():
    for dirpath, _, filenames in os.walk(JSON_PATH):
        for file in filenames:
            if file.endswith('.json'):  # Only process .json files
                json_file_path = os.path.join(dirpath, file)
                try:
                    with open(json_file_path, 'r') as f:
                        json_data = json.load(f)
                    json_data['source_file'] = file
                    json_data['source_directory'] = dirpath

                    data_list.append(json_data)

                except json.JSONDecodeError:
                    print(f"Error reading JSON file: {json_file_path}")

    df = pd.json_normalize(data_list)
    df.columns = [col.replace("landmarks.", "") for col in df.columns]
    df.columns = [col.replace('.', '_') for col in df.columns]
    for columns in df.columns:
        print(columns)
    df.to_csv(JSON_PATH + '.csv', index=False)

# if you already did make a file, it wont allow you to draw the hand landmarks again, so youre good.
if not os.path.exists(JSON_PATH):
    os.makedirs(JSON_PATH, exist_ok=True)

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
                   continue

               image_height, image_width, _ = color_image.shape

               rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
               results = hands.process(rgb_image)

               if results.multi_hand_landmarks:
                   hand_landmarks = results.multi_hand_landmarks[0]
                   landmarks_dict = {}

                   for idd, landmark in enumerate(hand_landmarks.landmark):
                       landmark_name = mp_hands.HandLandmark(idd).name
                       landmarks_dict[landmark_name] = {
                           "x": landmark.x,
                           "y": landmark.y,
                           'z': landmark.z
                       }

                   output_class_folder = os.path.join(JSON_PATH, GESTURE_CLASS)
                   os.makedirs(output_class_folder, exist_ok=True)

                   output_file = os.path.join(output_class_folder, f'landmarks_{file_name.replace(".jpg", ".json")}')
                   with open(output_file, 'w') as f:
                       json.dump({
                           "gesture_class": GESTURE_CLASS,
                           "landmarks": landmarks_dict},
                           f, indent=4)
                   print(f"Saved landmarks for {file_name}")

               if not results.multi_hand_landmarks:
                   print(f"Failed to detect hand landmarks in {file_name}")

option = input('want to view the items first or just save them? (v / s)')

if option.lower() == 'v':
    for gesture_class in os.listdir(JSON_PATH):
       gesture_folder = os.path.join(JSON_PATH, gesture_class)
       if not os.path.isdir(gesture_folder):
           continue


       for json_file in os.listdir(gesture_folder):
           if not json_file.endswith('.json'):
               continue


           json_path = os.path.join(gesture_folder, json_file)

           key = cv2.waitKey(25)
           if key == ord('f'):
               cv2.destroyAllWindows()
               break
           else:
               cv2.waitKey(0)

           with open(json_path, 'r') as f:
               data = json.load(f)

           image_name = json_file.replace("landmarks_", "").replace(".json", ".jpg")
           image_path = os.path.join(MAIN_PATH, gesture_class, image_name)

           image = cv2.imread(image_path)
           if image is None:
               print(f"Error loading image {image_name}")
               continue

           image_height, image_width, _ = image.shape
           draw_landmarks(image, data["landmarks"], image_width, image_height)

           cv2.namedWindow(f"{gesture_class} - {image_name}", cv2.WINDOW_NORMAL)
           cv2.resizeWindow(f"{gesture_class} - {image_name}", 600, 600)

           # Display the image
           cv2.imshow(f"{gesture_class} - {image_name}", image)


    good = input('does everything look good? are you ready to save to csv?(y / n)')
    if good.lower() == 'y':
        save_to_df()
    else:
        quit()

elif option.lower() == 's':
    save_to_df()

cv2.destroyAllWindows()  # Close all OpenCV windows
