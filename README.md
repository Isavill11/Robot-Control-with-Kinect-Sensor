# **Gesture-Based Robotic Arm Control using Kinect Sensor**


## Overview
This project is a work in progress and is part of my undergraduate research at Texas A&M International University. The goal is to create a cost-effective device to facilitate safe human-robot collaboration in manufacturing facilities. The system uses a Kinect One sensor to collect and process video data in real-time to classify objects and hand gestures to control an Intellitek robotic arm. This project integrates computer vision, machine learning, and robotics. 

## Features
* **Live Data Capture:** Uses OpenCV to collect video data from the Kinect sensor.
* **Object Detection:** Utilizes a YOLOV8 model for real-time object detection and tracking.
* **Hand Tracking & Gesture Recognition:** Applies Scikit-Learn and Mediapipe Holistic models for hand tracking and gesture classification.
* **Color Detection:** Uses OpenCV for color recognition color, signaling to the program that they are an authorized user and are, therefore, able to control the robotic arm. 
* **Robotic Arm Control:** Translates recognized gestures into commands for a robotic arm.
* **Data Preprocessing:** Captures, processes, and creates datasets for training object and gesture models.



## Tech needed:
- **Hardware:** Kinect Sensor
- **Software:**
  - Scorebase Robotic Control Software
  - Matlab

## Installation
To set up the environment, you first need to make sure you have downloaded *Kinect for Windows SDK 2.0*. Then, install the above libraries and PyKinect2024 to access the Kinect sensor using Python. 
```bash
pip install opencv-python numpy pandas scikit-learn
```
You may also need to install PyKinect and other dependencies manually.

## Usage
WIP
  
6. **Initiate Robotic Arm Control:** Using Matlab.Engine, call function from matlab to control the Robotic Arm. 

## Currently Developing/Improving
- GUI for custom data collection, model training, and live monitoring of the robotic arm
- Implement real-time feedback for gesture corrections.
- Once implemented, optimize robotic arm control response time.

## Contributors
**Isabella Villarreal**  

**Karina Silva**

**Frida Castilla**

**Alessandra Vela**

Undergraduate Researchers at Texas A&M International University


