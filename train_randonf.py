import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

TEST_SPLIT = 0.3     
NUM_TREES = 250         
RANDOM_STATE = 42       


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

# 3. Scale 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Train
model = RandomForestClassifier(n_estimators=NUM_TREES, random_state=RANDOM_STATE)
model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\nRandom Forest Accuracy: {acc*100:.2f}%")
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=classes, yticklabels=classes)
plt.title(f"Confusion Matrix (Acc: {acc*100:.1f}%)")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Plot 2: Feature Importance (Top 10)
plt.figure(figsize=(8,6))
importances = model.feature_importances_
# Reshape to average over time (130 steps -> 33 features)
imp_reshaped = importances.reshape(130, 33).mean(axis=0)
indices = np.argsort(imp_reshaped)[::-1][:10] # Top 10

feat_names = ([f"MFCC_{i+1}" for i in range(13)] + 
              [f"Chroma_{i+1}" for i in range(12)] + 
              [f"Contrast_{i+1}" for i in range(7)] + 
              ["Bandwidth"])
top_names = [feat_names[i] for i in indices]

plt.bar(range(10), imp_reshaped[indices], color='green')
plt.xticks(range(10), top_names, rotation=45, ha='right')
plt.title("Top 10 Important Features")
plt.show()
''' Our baseline experiments using Random Forest and SVM achieved a maximum accuracy of 40%. 
This limited performance suggests that flattening the temporal audio features into a 1D vector destroys critical time-dependent patterns. 
Therefore, a Deep Learning approach (CNN/LSTM) is necessary to capture the spatial and temporal structure of the audio. '''

# 0.2 100(Trees) = 38.23%, 0.2 150(Trees) = 39.34%, 0.2 200(Trees) = 41.00%, 0.2 250(Trees) = 40.72%
# 0.3 100(Trees) = 37.64%, 0.3 150(Trees) = 38.56%, 0.3 200(Trees) = 38.75%, 0.3 250(Trees) = 38.56%