import ctypes
import sys
import time
import numpy as np
import open3dlite as o3d
from pykinect2024 import PyKinect2024
from pykinect2024.PyKinectRuntime import PyKinect2024
from pykinect2024.PyKinectRuntime import PyKinectRuntime


class KinectDepthReader:
    def __init__(self):
        # Open Kinect sensor
        self.kinect = PyKinectRuntime.PyKinectV2().KinectSensor()
        self.kinect.open()

        # Open depth frame reader
        self.depth_frame_reader = self.kinect.depth_frame_source.open_reader()
        self.depth_frame_reader.frame_arrived += self.on_depth_frame

    def on_depth_frame(self, sender, args):
        depth_frame = args.frame_reference.acquire_frame()
        if depth_frame is not None:
            # Convert the depth frame to a numpy array
            depth_data = np.array(depth_frame.depth_pixels, dtype=np.uint16).reshape(
                (depth_frame.frame_desc.height, depth_frame.frame_desc.width))
            # Pass the depth data to Open3D
            self.process_depth_data(depth_data)

    def process_depth_data(self, depth_data):
        # Create an Open3D depth image from the numpy array
        depth_image = o3d.geometry.Image(depth_data)

        # Convert depth image to point cloud using a pinhole camera model
        # Set the intrinsic parameters (you may need to adjust these for your specific Kinect model)
        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        intrinsic.set_intrinsics(512, 424, 365.0, 365.0, 256.0, 212.0)

        # Create point cloud from depth image
        pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_image, intrinsic)

        # Visualize point cloud
        o3d.visualization.draw_geometries([pcd])

    def start(self):
        print("Press Enter to exit...")
        while True:
            time.sleep(0.1)

    def cleanup(self):
        self.depth_frame_reader.stop()


if __name__ == "__main__":
    kinect_reader = KinectDepthReader()
    kinect_reader.start()
    kinect_reader.cleanup()
