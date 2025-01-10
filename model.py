import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
from tensorflow.python.keras import *
import matplotlib.pyplot as plt
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from numpy import expand_dims

import numpy as np
import pandas as pd
import ultralytics
from ultralytics import *
from collections import defaultdict
import supervision as sv
import yaml
import random
import os
import shutil
from collections import Counter
from ultralytics import YOLO


def split_dataset(base_path, train_ratio=0.8, val_ratio=0.2, background_dir='./backgrounds'):
    # Create paths for images and labels
    images_path = os.path.join(base_path, 'images')
    labels_path = os.path.join(base_path, 'labels')

    # Create train, val, and test directories
    for set_type in ['train', 'val']:
        for content_type in ['images', 'labels']:
            os.makedirs(os.path.join(base_path, set_type, content_type), exist_ok=True)
#os. make directories.

    # Get all image filenames
    all_files = [f for f in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, f))]

    random.shuffle(all_files)

    # Calculate split indices
    total_files = len(all_files)
    train_end = int(train_ratio * total_files)
    val_end = train_end + int(val_ratio * total_files)

    # Split files
    train_files = all_files[:train_end]
    val_files = all_files[train_end:val_end]

    # Function to copy files
    def copy_files(files, set_type):
        for file in files:
            # Copy image
            shutil.copy(os.path.join(images_path, file), os.path.join(base_path, set_type, 'images'))
            # Copy corresponding label if it exists and is not empty
            label_file = file.rsplit('.', 1)[0] + '.txt'
            shutil.copy(os.path.join(labels_path, label_file), os.path.join(base_path, set_type, 'labels'))

    # Copy files to respective directories
    copy_files(train_files, 'train')
    copy_files(val_files, 'val')

#split the dataset into corresponding images and labels
split_dataset('C:/Users/Administrator/PycharmProjects/Kinect_body_tracking_v2/data_for_yolo')

class_count = Counter()
label_path = 'C:/Users/Administrator/PycharmProjects/Kinect_body_tracking_v2/data_for_yolo/labels'

## counts all the classes in the database and assigns then a number
for filename in os.listdir(label_path):
  label_file = os.path.join(label_path, filename)
  if os.path.isfile(label_file):
    with open(label_file, 'r') as f:
      lines = f.readlines()
      class_count.update([line.split()[0] for line in lines])
class_counts = dict(class_count)
print(class_count) # zero is a person, 1 is a robot.

# assigns the names to the number classes
class_names = ['person', 'robot']
class_counts_names = {class_names[int(class_id)]: count for class_id, count in class_counts.items()}

classes = list(class_counts_names.keys())
counts = list(class_counts_names.values())

## data visualization by creating a bar chart

# colors = ['Blue', 'Green']
# plt.bar(classes, counts, color= colors, width=.35)
#
# plt.xlabel("Classes")
# plt.ylabel("Number of Instances")
# plt.title("Class Distribution")
# plt.show()


###Model creation

datass = {"path": "C:/Users/Administrator/PycharmProjects/Kinect_body_tracking_v2/data_for_yolo",
        "train": "C:/Users/Administrator/PycharmProjects/Kinect_body_tracking_v2/data_for_yolo/train",
        "val": "C:/Users/Administrator/PycharmProjects/Kinect_body_tracking_v2/data_for_yolo/val",
        "nc": len(classes),
        "names":classes
}
with open("data.yaml", "w") as f:   ##this is where u rename the .yaml file name.
  yaml.dump(datass, f)

model = YOLO("yolov8n.pt", task="detect")

results = model.train(data='C:/Users/Administrator/PycharmProjects/Kinect_body_tracking_v2/data.yaml', epochs=80)  #put the name of the .yaml file


## predict new images
# image = '/content/45.jpg'
# model.predict(image, save=True, imgsz=640)



## load another model and transfer weights to imporve model accuracy

# model2 = YOLO("yolov8x.yaml").load('yolov8x.pt')#build off the best weights from the previous model apparently
# model2.train(data='/content/data.yaml', epochs=3)