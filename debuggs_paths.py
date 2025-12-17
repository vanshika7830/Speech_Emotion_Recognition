import os
import numpy as np
DATASET_PATH = "dataset_raw"

print(f"--- DEBUGGING PATH: {os.path.abspath(DATASET_PATH)} ---")

if not os.path.exists(DATASET_PATH):
    print("ERROR: The folder 'dataset_raw' does not exist here.")
    print("   Make sure you are running this script in the same folder as dataset_raw.")
    exit()

found_files = 0
found_folders = 0

for root, dirs, files in os.walk(DATASET_PATH):
    # Skip the root folder itself
    if root == DATASET_PATH:
        if len(files) > 0:
            print(f"WARNING: Found {len(files)} files in the root folder!")
            print(f"   These are being IGNORED. Move them into subfolders (e.g., '{DATASET_PATH}/happy/').")
            print(f"   Example: {files[0]}")
        continue

  
    folder_name = os.path.basename(root)
    print(f"Found Class Folder: '{folder_name}' containing {len(files)} files.")
    
    # Check for valid extensions
    audio_files = [f for f in files if f.endswith(('.mp3', '.wav'))]
    if len(audio_files) == 0 and len(files) > 0:
        print(f"   ERROR: Folder contains files but no .mp3 or .wav!")
        print(f"   Example file: {files[0]}")
    
    found_files += len(audio_files)
    found_folders += 1

print("\n--- SUMMARY ---")
if found_files == 0:
    print("FAILURE: Python found 0 valid audio files.")
else:
    print(f"SUCCESS: Found {found_files} audio files in {found_folders} categories.")
    print("You can run preprocess.py now.")

# Check if X_data has data or not...

# Load the file
X = np.load("X_data.npy")

# Print the 3 Parameters (Dimensions)
print(f"Dataset Shape: {X.shape}")

# LOGIC CHECK
if X.ndim == 3:
    print("  SUCCESS: You have 3 dimensions!")
    print(f"   1. Files:      {X.shape[0]}")
    print(f"   2. Time Steps: {X.shape[1]}")
    print(f"   3. Features:   {X.shape[2]}")
elif X.ndim == 1 and X.shape[0] == 0:
    print("  FAILURE: The file is empty.")
    print("   This means preprocess.py found 0 audio files.")
    print("   CHECK: Did you put your audio files inside subfolders?")
    print("   Correct:  dataset_raw/happy/laugh.mp3")
    print("   Wrong:    dataset_raw/laugh.mp3")
else:
    print(" WARNING: You have data, but the shape is wrong.")