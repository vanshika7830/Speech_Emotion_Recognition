import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# ================= CONFIGURATION =================
TEST_SPLIT = 0.25       # Try 0.2, 0.25, 0.3
NUM_TREES = 100         # Try 50, 100, 200
RANDOM_STATE = 42       # Keep this fixed for reproducible results
# =================================================

print("--- TRAINING RANDOM FOREST ---")

# 1. Load & Flatten Data
X = np.load("X_data.npy")
y = np.load("y_data.npy")
classes = np.load("classes.npy")

# Flatten (Files, Time, Feat) -> (Files, Flat_Vector)
X_flat = X.reshape(X.shape[0], -1)

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(
    X_flat, y, test_size=TEST_SPLIT, random_state=RANDOM_STATE
)

# 3. Scale (Optional for RF, but good practice)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Train
model = RandomForestClassifier(n_estimators=NUM_TREES, random_state=RANDOM_STATE)
model.fit(X_train, y_train)

# 5. Evaluate
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

print(f"\n✅ Random Forest Accuracy: {acc*100:.2f}%")
print(f"   (Split: {1-TEST_SPLIT}/{TEST_SPLIT}, Trees: {NUM_TREES})")
print("\nDetailed Report:")
print(classification_report(y_test, preds, target_names=classes))


''' Our baseline experiments using Random Forest and SVM achieved a maximum accuracy of 40%. 
This limited performance suggests that flattening the temporal audio features into a 1D vector destroys critical time-dependent patterns. 
Therefore, a Deep Learning approach (CNN/LSTM) is necessary to capture the spatial and temporal structure of the audio. '''