from ultralytics import YOLO
import joblib
import mediapipe as mp
from keras._tf_keras.keras.models import load_model



obj_model = YOLO("C:/Users/Administrator/PycharmProjects/Kinect_body_tracking_v2/runs/detect/train/weights/best.pt")
gesture_model = load_model('gesture_model.keras')
scaler = joblib.load('_scaler2.pkl')
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hand = mp_hands.Hands()

