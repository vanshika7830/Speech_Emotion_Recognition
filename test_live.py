import numpy as np
import librosa
import tensorflow as tf
import noisereduce as nr


MODEL_PATH = "best_cnn_model.h5"
TEST_FILE = "dataset_raw/angry/argument_5610_erny2_wav.mp3" # <--- Your file
CLASSES = np.load("classes.npy")


def predict_debug(file_path):
    print(f"🎤 Analyzing: {file_path}")
    
    # 1. LOAD
    signal, sr = librosa.load(file_path, sr=22050)
    
    # --- FIX 1: NORMALIZE VOLUME (Crucial!) ---
    # This makes the volume range 0 to 1, matching standard training data
    signal = librosa.util.normalize(signal)
    
    # 2. PREPROCESS
    # Denoise (Lightly)
    try:
        signal = nr.reduce_noise(y=signal, sr=sr, prop_decrease=0.5)
    except:
        pass # Skip if audio is too short for noise reduction

    # Fix Length (3.0s)
    target_len = 22050 * 3
    if len(signal) >= target_len:
        signal = signal[:target_len]
    else:
        signal = np.pad(signal, (0, target_len - len(signal)), 'constant')

    # 3. EXTRACT FEATURES (Must match training)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)
    chroma = librosa.feature.chroma_stft(y=signal, sr=sr, n_fft=2048, hop_length=512)
    contrast = librosa.feature.spectral_contrast(y=signal, sr=sr, n_fft=2048, hop_length=512)
    bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sr, n_fft=2048, hop_length=512)
    
    # Stack & Transpose
    feat = np.concatenate([mfcc, chroma, contrast, bandwidth], axis=0).T
    
    # Safe Reshape
    if feat.shape[0] > 130: feat = feat[:130, :]
    elif feat.shape[0] < 130: 
        feat = np.pad(feat, ((0, 130 - feat.shape[0]), (0, 0)), 'constant')
        
    feat = feat.reshape(1, 130, 33, 1)

    # 4. PREDICT
    model = tf.keras.models.load_model(MODEL_PATH)
    prediction = model.predict(feat, verbose=0)[0] # Get the first (and only) result

    # 5. SHOW TOP 3 GUESSES
    # Sort indices from high to low
    sorted_indices = np.argsort(prediction)[::-1]
    
    print("\n" + "="*30)
    print("MODEL THINKING PROCESS:")
    print("="*30)
    
    for i in range(3): # Show Top 3
        idx = sorted_indices[i]
        label = CLASSES[idx]
        prob = prediction[idx] * 100
        print(f"{i+1}. {label.upper()}: {prob:.2f}%")
        
    print("="*30)

    # 6. SANITY CHECK
    if prediction[sorted_indices[0]] < 0.50:
        print("\nWARNING: Low Confidence (<50%).")
        print("   The model is confused. Try recording clearly without background noise.")

predict_debug(TEST_FILE)