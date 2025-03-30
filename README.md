#**Kinect Gesture Recognition and Robotic Arm Control!!!**


## Overview
This project is a work in progress and is part of my undergraduate research at Texas A&M International University. The goal is to create a cost-effective device to facilitate safe human-robot collaboration in manufacturing facilities. The system uses a Kinect One sensor to collect and process video data in real-time to classify objects and hand gestures to control an Intellitek robotic arm. This project integrates computer vision, machine learning, and robotics. 

## Features
* **Live Data Capture:** Uses OpenCV to collect video data from the Kinect sensor.
* **Object Detection:** Utilizes a YOLOV8 model for real-time object detection and tracking.
* **Hand Tracking & Gesture Recognition:** Applies Scikit-Learn and Mediapipe Holistic models for hand tracking and gesture classification.
* **Color Detection:** Uses OpenCV for color recognition color, signaling to the program that they are an authorized user and are, therefore, able to control the robotic arm. 
* **Robotic Arm Control:** Translates recognized gestures into commands for a robotic arm.
* **Data Preprocessing:** Captures, processes, and creates datasets for training object and gesture models.

## Technologies Used
* **Hardware:** Kinect Sensor
* **Softwares:**
*  Scorebase Robotic Control Software
*  VS Code
*  Matlab
* **Libraries & Frameworks:**
*  OpenCV
*  Scikit-Learn
*  YOLOV8
*  label-studio
*  PyKinect2024
*  matlab.engine
*  NumPy, Pandas, Matplotlib, etc.

## Installation
To set up the environment, you first need to make sure you have downloaded *Kinect for Windows SDK 2.0*. Then, install the above libraries and PyKinect2024 to access the Kinect sensor using Python. 
```bash
pip install opencv-python numpy pandas scikit-learn
```
You may also need to install PyKinect and other dependencies manually.

## Usage
Depending on the date, I probably have not cleaned up my code. Functional, yes; pretty, not yet.
1. **Run the Kinect Sensor:** Ensure the Kinect is connected to your PC.
2. **Start Data Capture:** Run the script to collect real-time data.
   ```bash
   python capture_data.py
   ```
3. **Preprocess Data:** Convert raw data into a structured dataset.
   ```bash
   python preprocess_data.py
   ```
4. **Train the Model:** Train the gesture classification model.
   ```bash
   python train_model.py
   ```
5. **Deploy Gesture Recognition:** Use the trained model for real-time detection.
   ```bash
   python gesture_recognition.py
   ```

## Future Improvements
- Create functions for the robotic arm to command it. 
- Improve model accuracies with more training data.
- Implement real-time feedback for gesture corrections.
- Once implemented, optimize robotic arm control response time.
- Create desktop app for easy usage.
- Clean up code. 

## Contributors
**Isabella Villarreal**  
Undergraduate Researcher, Texas A&M International University

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
For any questions or collaboration inquiries, feel free to reach out via [LinkedIn](https://www.linkedin.com/in/isabellaavillarreal/).



