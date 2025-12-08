import os
import librosa
import numpy as np
import noisereduce as nr  # pip install noisereduce

# ================= CONFIGURATION =================
DATASET_PATH = "dataset_raw"
SAMPLE_RATE = 22050
DURATION = 3  # Seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
# =================================================

def preprocess_dataset(dataset_path):
    data = {
        "mapping": [],
        "features": [],
        "labels": []
    }
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"Error: Folder '{dataset_path}' not found.")
        return

    print(f"--- STARTING ADVANCED PREPROCESSING ---")
    print(f"Target: {DURATION}s @ {SAMPLE_RATE}Hz")
    
    # Loop through emotion folders
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        
        # Ensure we are in a subfolder (not root)
        if dirpath is not dataset_path:
            label = dirpath.split(os.sep)[-1]
            data["mapping"].append(label)
            print(f"\n📂 Processing Class: '{label}'")
            
            for f in filenames:
                file_path = os.path.join(dirpath, f)
                
                # Skip non-audio files
                if not f.lower().endswith(('.mp3', '.wav')):
                    continue

                try:
                    # 1. LOAD AUDIO & RESAMPLE
                    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                    
                    # 2. NOISE CANCELLATION (Spectral Gating)
                    # Removes background static/hiss
                    signal = nr.reduce_noise(y=signal, sr=sr, prop_decrease=0.8)
                    
                    # 3. ENFORCE FIXED DURATION (3 Seconds)
                    if len(signal) >= SAMPLES_PER_TRACK:
                        # TRUNCATE (Cut off the end)
                        signal = signal[:SAMPLES_PER_TRACK]
                    else:
                        # PAD (Add silence to the end)
                        signal = np.pad(signal, (0, SAMPLES_PER_TRACK - len(signal)), 'constant')
                    
                    # 4. ADVANCED FEATURE EXTRACTION
                    # We use same hop_length=512 so all features align in time
                    
                    # A. MFCC (Texture/Timbre) - 13 features
                    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)
                    
                    # B. Chroma (Pitch/Melody) - 12 features
                    chroma = librosa.feature.chroma_stft(y=signal, sr=sr, n_fft=2048, hop_length=512)
                    
                    # C. Spectral Contrast (Roughness) - 7 features
                    contrast = librosa.feature.spectral_contrast(y=signal, sr=sr, n_fft=2048, hop_length=512)
                    
                    # D. Spectral Bandwidth (Sharpness) - 1 feature
                    bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sr, n_fft=2048, hop_length=512)
                    
                    # 5. STACK FEATURES
                    # Stack them vertically: 13 + 12 + 7 + 1 = 33 Features
                    combined = np.concatenate([mfcc, chroma, contrast, bandwidth], axis=0)
                    
                    # 6. TRANSPOSE
                    # Shape becomes (Time_Steps, Features) -> (130, 33)
                    # This is what the CNN expects (Rows=Time, Cols=Features)
                    combined = combined.T
                    
                    data["features"].append(combined.tolist())
                    data["labels"].append(i-1)
                    
                    print(f"  -> Processed: {f}", end="\r")
                    
                except Exception as e:
                    print(f"  ❌ Corrupt file {f}: {e}")

    # Convert to Numpy Arrays
    X = np.array(data["features"])
    y = np.array(data["labels"])
    mapping = data["mapping"]
    
    print("\n\n=========================================")
    print("       PREPROCESSING COMPLETE")
    print("=========================================")
    print(f"Total Files Processed: {len(X)}")
    print(f"Feature Matrix Shape:  {X.shape}")
    print(f"   - Time Steps:       {X.shape[1]} (Should be ~130)")
    print(f"   - Total Features:   {X.shape[2]} (Should be 33)")
    
    # Save to disk
    np.save("X_data.npy", X)
    np.save("y_data.npy", y)
    np.save("classes.npy", mapping)
    print("✔ Saved 'X_data.npy', 'y_data.npy', 'classes.npy'")

if __name__ == "__main__":
    preprocess_dataset(DATASET_PATH)