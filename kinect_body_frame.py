import pykinect2024 as pk
from pykinect2024 import PyKinectRuntime, PyKinect2024
import pygame
import ctypes
import numpy as np
import time


pygame.init()
screen_width, screen_height = 640, 480
screen = pygame.display.set_mode((screen_width, screen_height), pygame.HWSURFACE | pygame.DOUBLEBUF)

pygame.display.set_caption("Kinect Body Frame Visualization")
clock = pygame.time.Clock()

kinect = PyKinectRuntime.PyKinectRuntime(PyKinectRuntime.FrameSourceTypes_Body | PyKinectRuntime.FrameSourceTypes_Color)


joint_points = [pk.PyKinect2024.JointType_Head,
                pk.PyKinect2024.JointType_Neck,
                pk.PyKinect2024.JointType_SpineShoulder,
                pk.PyKinect2024.JointType_SpineMid,
                pk.PyKinect2024.JointType_SpineBase,
                pk.PyKinect2024.JointType_ShoulderLeft,
                pk.PyKinect2024.JointType_ElbowLeft,
                pk.PyKinect2024.JointType_WristLeft,
                pk.PyKinect2024.JointType_HandLeft,
                pk.PyKinect2024.JointType_ShoulderRight,
                pk.PyKinect2024.JointType_ElbowRight,
                pk.PyKinect2024.JointType_WristRight,
                pk.PyKinect2024.JointType_HandRight,
                pk.PyKinect2024.JointType_HipLeft,
                pk.PyKinect2024.JointType_KneeLeft,
                pk.PyKinect2024.JointType_AnkleLeft,
                pk.PyKinect2024.JointType_FootLeft,
                pk.PyKinect2024.JointType_HipRight,
                pk.PyKinect2024.JointType_KneeRight,
                pk.PyKinect2024.JointType_AnkleRight,
                pk.PyKinect2024.JointType_FootRight]

def draw_body(joints, joint_points):
    ### draw the skeleton based on joint positions
    for joint in joint_points:
        joint_state = joints[joint].TrackingState
        if joint_state == pk.PyKinect2024.TrackingState_Tracked:
            pos = kinect.body_joint_to_color_space(joints)[joint]
            pygame.draw.circle(screen, (0,255,0), (int(pos.x), int(pos.y)), 5)

def draw_color_frame():
    frame = kinect.get_last_color_frame()
    if frame is None:
        return

    color_image = frame.reshape((1080, 1920, 4))
    surface = pygame.image.frombuffer(color_image.tobytes(), (1920, 1080), "BGRA")
    resized_surface = pygame.transform.scale(surface, (screen_width, screen_height))
    screen.blit(resized_surface, (0, 0))


running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    draw_color_frame()

    if kinect.has_new_body_frame():
        bodies = kinect.get_last_body_frame()
        if bodies is not None:
            for i in range(0, kinect.max_body_count):
                body = bodies.bodies[i]
                if body.is_tracked:
                    joints = body.joints
                    draw_body(joints, joint_points)

    pygame.display.flip()
    clock.tick(30)


kinect.close()
pygame.quit()