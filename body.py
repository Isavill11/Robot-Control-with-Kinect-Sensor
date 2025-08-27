import os
import cv2
import numpy as np
from google.protobuf.json_format import MessageToDict

OBJ_CLASSES = {0: ["Person", (204,0,0)], #red
           1: ["Robot", (0, 128, 255)],} #blue

COLOR_TO_TRACK = {"lower_range": np.array([10, 150, 150]),
                  "upper_range": np.array([25, 255, 255])} # the rgb range for whatever color
threshold = 40  ##min number of pixels needed to determine if the colors present.




class UserHand(): 
    
    LANDMARK_COLOR = (0, 255, 0)  # Green
    CONNECTION_COLOR = (255, 0, 0)  # Blue
    GESTURE_HOLD_TIME = 3 #seconds
    COOLDOWN_DURATION = 5 #seconds 

    def __init__(self, hand_label="Right", gesture_folder="gestures"):
        self.hand_label = hand_label
        self.gesture = None
        self.current_position = []
        self.last_trigger_time = 0
        self.gesture_timer = None
        self.gesture_folder = gesture_folder
        self.GESTURE_CLASSES = self._load_gestures()

#TODO: ensure that when the user adds a new gesture, its added to the working folder path and a folder called 'gestures'
    def _load_gestures(self): ### ensure that when 
        folder_path = os.path.join(os.getcwd(), self.gesture_folder)
        if os.path.exists(folder_path):
            return [g for g in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, g))]
        return []

    
    def track(self, frame, roi, hand, scaler, gesture_model, mp_drawing, mp_hands):
        
        predicted_gesture = None

        try:
            roi_width, roi_height, _ = roi.shape
            rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            results_gesture = hand.process(rgb_roi)

            if results_gesture.multi_hand_landmarks and results_gesture.multi_handedness:
                roi_mask = np.zeros_like(roi)

                for i, hand_landmarks in enumerate(results_gesture.multi_hand_landmarks):
                    label = MessageToDict(results_gesture.multi_handedness[i])['classification'][0]['label']
                    if label == self.hand_label:
                        new_row = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark],
                                        dtype=np.float32).flatten().reshape(1, -1)
                        new_input_scaled = scaler.transform(new_row)

                        y_pred = gesture_model.predict(new_input_scaled)
                        y_pred_class = np.argmax(y_pred, axis=1)[0]
                        predicted_gesture = self.GESTURE_CLASSES[y_pred_class] if y_pred_class < len(self.GESTURE_CLASSES) else None

                        # Draw landmarks + overlay text
                        for idx, landmark in enumerate(hand_landmarks.landmark):
                            x, y = int(landmark.x * roi_width), int(landmark.y * roi_height)
                            if idx == 0 and predicted_gesture:
                                cv2.putText(frame, predicted_gesture, (x - 20, y - 20),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                        mp_drawing.draw_landmarks(roi_mask, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Overlay landmarks on original frame
                frame = cv2.addWeighted(frame, 1, roi_mask, 1, 0)

            return predicted_gesture

        except Exception as e:
            print(f"Error tracking joints: {e}")
            return None
        


class UserBody:
    def __init__(self):
        self.left_hand = UserHand("Left")
        self.right_hand = UserHand("Right")

    def track_body(self, frame, roi, hand, scaler, gesture_model, mp_drawing, mp_hands):
        left_gesture = self.left_hand.track(frame, roi, hand, scaler, gesture_model, mp_drawing, mp_hands)
        right_gesture = self.right_hand.track(frame, roi, hand, scaler, gesture_model, mp_drawing, mp_hands)
        return left_gesture, right_gesture