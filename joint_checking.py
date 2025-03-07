

import json
import os
import cv2
import mediapipe as mp

working_dir = os.getcwd()
# Paths
MAIN_PATH = working_dir + '/hand_data_extended'
print(MAIN_PATH)
JSON_PATH = working_dir + '/hand_data_extended_from_mediapipe'  # Path where JSON files with landmarks are stored

# MediaPipe drawing utils
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Define colors
LANDMARK_COLOR = (0, 255, 0)  # Green
CONNECTION_COLOR = (255, 0, 0)  # Blue



# Function to visualize landmarks on an image
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


# Process all JSON files
for gesture_class in os.listdir(JSON_PATH):
   gesture_folder = os.path.join(JSON_PATH, gesture_class)
   if not os.path.isdir(gesture_folder):
       continue


   for json_file in os.listdir(gesture_folder):
       if not json_file.endswith('.json'):
           continue


       json_path = os.path.join(gesture_folder, json_file)


       # Load JSON data
       with open(json_path, 'r') as f:
           data = json.load(f)


       image_name = json_file.replace("landmarks_", "").replace(".json", ".jpg")
       image_path = os.path.join(MAIN_PATH, gesture_class, image_name)


       # Load the original image
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

       key = cv2.waitKey(25)
       if key == ord('q'):
           cv2.destroyAllWindows()
           quit()
       else:
           cv2.waitKey(0)

cv2.destroyAllWindows()  # Close all OpenCV windows
