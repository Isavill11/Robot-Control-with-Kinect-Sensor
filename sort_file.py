import os
import shutil



DIR_NAME_NEW = 'Gesture_Model'
WORKING_DIR = os.getcwd()
file_extention = ['.keras', '.pkl', '.h5']


if not os.path.exists(DIR_NAME_NEW):
    os.mkdir(DIR_NAME_NEW)


for filename in os.listdir(WORKING_DIR):
    for ext in file_extention:
        if filename.endswith(ext):
            shutil.move(os.path.join(WORKING_DIR, filename), os.path.join(DIR_NAME_NEW, filename))





