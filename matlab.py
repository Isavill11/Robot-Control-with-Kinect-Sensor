
import matlabengine

m = matlab.engine.start_matlab('MATLAB_8178')













### SORRRY I NEEDED TO CLEAN UP OUR FILES D:


# import os
# import shutil
#
#
# MAIN_DIR = os.getcwd()
# DIR_FOR_OBJECT_DETECTION = 'All_Yolo_Data'
# DIR_FOR_GESTURE_REC = 'Gesture_Recognition_Data'
# RAW_DATA_DIR = 'Raw_Data'
#
# od_dir = os.path.join(MAIN_DIR, DIR_FOR_OBJECT_DETECTION)
# gr_dir = os.path.join(MAIN_DIR, DIR_FOR_GESTURE_REC)
#
# if not os.path.exists(od_dir):
#     os.makedirs(od_dir)
#
# if not os.path.exists(gr_dir):
#     os.makedirs(gr_dir)
#
#
# folders_to_move_OD = ['Annotated_Data_for_yolo', 'annotated_OD_data_for_yolo', 'object_data2', 'object_detection_data', 'data']
# folders_to_move_GR = ['data_for_gesture_rec', 'depth_color_data', 'failed_data', 'hand_data']
#
# for folder_name in folders_to_move_OD:
#     folder_path = os.path.join(MAIN_DIR, folder_name)
#     destination = od_dir
#
#     if os.path.exists(folder_path):
#         shutil.move(folder_path, destination)
#         print(f'Moved {folder_name} to {destination}')
#     else:
#         print(f'Folder {folder_name} not found.')
#
# for folder_name in folders_to_move_GR:
#     folder_path = os.path.join(MAIN_DIR, folder_name)
#     destination = gr_dir
#
#     if os.path.exists(folder_path):
#         shutil.move(folder_path, destination)
#         print(f'Moved {folder_name} to {destination}')
#     else:
#         print(f'Folder {folder_name} not found.')