import os
import shutil
import random

# Lists of file paths
unknown_files = list(range(1,21))  # Add your unknown file paths here
keyword_files = list(range(21,41))  # Add your keyword file paths here

# Base directory where folders should be created
base_dir = r'D:\keyword_spotting\data'  # Replace this with your desired base directory path

# Distribution percentages
test_split = 0.1
validation_split = 0.1
training_split = 1 - (test_split + validation_split)

# Folder names
main_folders = ['training', 'testing', 'validation']
subfolders = ['keyword', 'unknown']

# Create folders and subfolders in the specified base directory
for folder in main_folders:
    for subfolder in subfolders:
        os.makedirs(os.path.join(base_dir, folder, subfolder), exist_ok=True)

# Function to split and move files
def distribute_files(files, folder_name):
    random.shuffle(files)  # Shuffle to ensure randomness

    # Determine number of files per split
    test_count = int(len(files) * test_split)
    validation_count = int(len(files) * validation_split)
    training_count = len(files) - test_count - validation_count

    # Split files into training, testing, and validation
    test_files = files[:test_count]
    validation_files = files[test_count:test_count + validation_count]
    training_files = files[test_count + validation_count:]

    # Move files to respective folders in the base directory
    for file in test_files:
        shutil.copy2(file, os.path.join(base_dir, 'testing', folder_name, os.path.basename(file)))
    for file in validation_files:
        shutil.copy2(file, os.path.join(base_dir, 'validation', folder_name, os.path.basename(file)))
    for file in training_files:
        shutil.copy2(file, os.path.join(base_dir, 'training', folder_name, os.path.basename(file)))

# Distribute the files
distribute_files(keyword_files, 'keyword')
distribute_files(unknown_files, 'unknown')

print("Files have been moved successfully.")
