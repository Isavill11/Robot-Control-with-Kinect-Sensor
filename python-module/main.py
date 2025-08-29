from kinect import MyKinect
from body import *
import cv2
import time
from Deep_Learning.control_scorbot import execute_matlab_commands
from body import get_latest_frame, check_for_color, joint_tracking
from obj_ges_models import *


kinect = MyKinect()



while True:
    try:
        newest_frame = get_latest_frame()
        image_width, image_height, _  = newest_frame.shape
        results_obj = obj_model.predict(newest_frame, verbose = False)

        for result in results_obj: #just gives us access to all the data.
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0]) #bounding boxes coordinates
                if x1 < 0 or y1 < 0 or x2 > newest_frame.shape[1] or y2 > newest_frame.shape[0]:
                    ## make this so it still displays but cuts off at the edge of the frame.
                    print("Invalid bounding box coordinates")
                else:
                    roi = newest_frame[y1:y2, x1:x2]

                confidence, obj_class = box.conf[0], int(box.cls[0])
                label = f"{OBJ_CLASSES[obj_class][0]} {confidence:.2f}" #class label and confidence level

                if confidence > 0.40:
                    cv2.rectangle(newest_frame, (x1, y1), (x2, y2), OBJ_CLASSES[obj_class][1])
                    cv2.putText(newest_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=OBJ_CLASSES[obj_class][1], thickness=2)

                    if check_for_color(roi, COLOR_TO_TRACK["lower_range"], COLOR_TO_TRACK["upper_range"]) and OBJ_CLASSES[obj_class][0] == "Person":
                        gesture = joint_tracking(newest_frame, roi)

                        if gesture:
                            if gesture == current_gesture:
                                if gesture_timer and time.time() - gesture_timer >= GESTURE_HOLD_TIME:
                                    if time.time() - last_trigger_time >= cooldown_duration:
                                        print(f"Gesture '{gesture}' held for 3 seconds. Triggering robotic arm.")
                                        execute_matlab_commands()
                                        last_trigger_time = time.time()
                                    else:
                                        print("Cooldown active. Gesture ignored.")
                                    gesture_timer = None
                                    current_gesture = None
                            else:
                                current_gesture = gesture
                                gesture_timer = time.time()
                        else:
                            current_gesture = None
                            gesture_timer = None
                        # print(f"color detected in bounding box:{x1},{y1},{x2},{y2}.")
                        # cv2.putText(newest_frame, "Authorized User!!", (x1+10, y1+50), cv2.FONT_HERSHEY_PLAIN, 2, color = (0, 255, 0), thickness=4)

        cv2.imshow("Object Tracking", newest_frame) ## this updates the screen so we can see the imaginary screen now.

    except AttributeError as e:
        print(f"Error: {e}. Cant capture frame right now.")
        continue

    if cv2.waitKey(33) & 0xFF == ord('q'):
        break

kinect.close()
cv2.destroyAllWindows()






