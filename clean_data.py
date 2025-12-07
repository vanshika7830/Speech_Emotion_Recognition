import os
import hashlib
import librosa
import numpy as np

# CONFIGURATION
DATASET_DIR = "dataset_raw"
MIN_DURATION = 0.5   # Seconds (Delete anything shorter)
MAX_DURATION = 6.0   # Seconds (Delete anything longer)
SILENCE_THRESHOLD = 0.005 # Amplitude threshold (Delete if quieter than this)

def calculate_hash(file_path):
    """Generates a digital fingerprint of the file."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def clean_dataset():
    print(f"--- Starting Deep Cleaning in '{DATASET_DIR}' ---")
    
    unique_hashes = {}
    files_removed = 0
    duplicates_removed = 0
    
    # Walk through all folders (happy, angry, etc.)
    for root, dirs, files in os.walk(DATASET_DIR):
        for filename in files:
            if not filename.endswith(('.mp3', '.wav')):
                continue
                
            file_path = os.path.join(root, filename)
            
            # --- STEP 1: REMOVE DUPLICATES ---
            file_hash = calculate_hash(file_path)
            
            if file_hash in unique_hashes:
                print(f"[DUPLICATE] Deleting {filename} (Same as {unique_hashes[file_hash]})")
                os.remove(file_path)
                duplicates_removed += 1
                continue # Skip to next file
            else:
                unique_hashes[file_hash] = filename

            # --- STEP 2: REMOVE BAD AUDIO (Silence/Length) ---
            try:
                # Load audio (fast load)
                y, sr = librosa.load(file_path, sr=None)
                duration = librosa.get_duration(y=y, sr=sr)
                
                # Check Duration
                if duration < MIN_DURATION or duration > MAX_DURATION:
                    print(f"[BAD LENGTH] Deleting {filename} ({duration:.2f}s)")
                    os.remove(file_path)
                    files_removed += 1
                    continue
                
                # Check for Silence (RMS Energy)
                rms = np.sqrt(np.mean(y**2))
                if rms < SILENCE_THRESHOLD:
                    print(f"[SILENCE] Deleting {filename} (Too quiet)")
                    os.remove(file_path)
                    files_removed += 1
                    continue
                    
            except Exception as e:
                print(f"[CORRUPT] Deleting {filename} (Cannot load)")
                os.remove(file_path)
                files_removed += 1

    print("\n--- CLEANING SUMMARY ---")
    print(f"✔ Duplicates Removed: {duplicates_removed}")
    print(f"✔ Bad Files Removed:  {files_removed}")
    print(f"✔ Total Remaining:    {len(unique_hashes)}")

if __name__ == "__main__":
    clean_dataset()