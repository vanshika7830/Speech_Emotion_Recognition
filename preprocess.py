import os
import librosa
import numpy as np
import noisereduce as nr  

# Configuration
DATASET_PATH = "dataset_raw"
SAMPLE_RATE = 22050
DURATION = 3 
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

def preprocess_dataset(dataset_path):
    data = {
        "mapping": [],
        "mfcc": [],
        "labels": []
    }
    
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        if dirpath is not dataset_path:
            label = dirpath.split("\\")[-1] 
            data["mapping"].append(label)
            print(f"\nProcessing: '{label}'")
            
            for f in filenames:
                file_path = os.path.join(dirpath, f)
                
                try:
                    # 1. LOAD AUDIO
                    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                    
                    # --- NEW STEP: NOISE CANCELLATION ---
                    # This reads the audio, guesses what is "noise" (usually the quiet parts),
                    # and subtracts it from the loud parts.
                    # prop_decrease=0.8 means "remove 80% of the noise" 
                    # (Avoiding 1.0 prevents the voice from sounding robotic)
                    signal = nr.reduce_noise(y=signal, sr=sr, prop_decrease=0.8)
                    # ------------------------------------

                    # 2. FORCE FIXED LENGTH (Padding/Truncating)
                    if len(signal) >= SAMPLES_PER_TRACK:
                        signal = signal[:SAMPLES_PER_TRACK]
                    else:
                        signal = np.pad(signal, (0, SAMPLES_PER_TRACK - len(signal)), 'constant')
                    
                    # 3. EXTRACT MFCC
                    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)
                    mfcc = mfcc.T
                    
                    data["mfcc"].append(mfcc.tolist())
                    data["labels"].append(i-1)
                    print(f"  -> {f} (Denoised)", end="\r")
                    
                except Exception as e:
                    print(f"  ! Error parsing {f}: {e}")

    X = np.array(data["mfcc"])
    y = np.array(data["labels"])
    mapping = data["mapping"]
    
    print("\n\n--- PREPROCESSING COMPLETE ---")
    print(f"Features (X) Shape: {X.shape}")
    print(f"Labels (Y) Shape: {y.shape}")
    
    np.save("X_data.npy", X)
    np.save("y_data.npy", y)
    np.save("classes.npy", mapping)
    print("✔ Saved X_data.npy (Cleaned) and y_data.npy")

if __name__ == "__main__":
    preprocess_dataset(DATASET_PATH)