
import os
import json
import pandas as pd

working_dir = os.getcwd()
# Define the path to the main JSON directory
JSON_PATH = working_dir+'/hand_data_xyz_coords'

data_list = []

# Recursively walk through the directory
# os walk outputs a tuple in the form:
# 1) current directory path being traversed, this can be 'main_dir' or 'main_dir/subfolder1', etc.
# 2) subdirectory names in the main path, and
# 3) the file names in the current directory path.
for dirpath, _, filenames in os.walk(JSON_PATH):
   for file in filenames:
       if file.endswith('.json'):  # Only process .json files
           json_file_path = os.path.join(dirpath, file)
           try:
               # Read the JSON file
               with open(json_file_path, 'r') as f:
                   json_data = json.load(f)

               # Add metadata (e.g., file name or directory) for tracking
               json_data['source_file'] = file
               json_data['source_directory'] = dirpath

               # Append the JSON data to our list
               data_list.append(json_data)

           except json.JSONDecodeError:
               print(f"Error reading JSON file: {json_file_path}")


df = pd.json_normalize(data_list)
df.columns = [col.replace("landmarks.", "") for col in df.columns]
df.columns = [col.replace('.', '_') for col in df.columns]
print(df)
for columns in df.columns:
   print(columns)
df.to_csv('hand_xyz_coordinates.csv', index=False)


