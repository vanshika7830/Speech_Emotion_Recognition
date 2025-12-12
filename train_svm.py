import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# ================= CONFIGURATION =================
TEST_SPLIT = 0.2  # Try 0.2, 0.3
KERNEL_TYPE = 'linear'     # Try 'linear', 'poly', 'rbf'
C_VALUE = 1.0           # Try 0.1, 1.0, 10.0 (Higher = stricter margin)
# =================================================

print("--- TRAINING SVM ---")

# 1. Load & Flatten
X = np.load("X_data.npy")
y = np.load("y_data.npy")
classes = np.load("classes.npy")

X_flat = X.reshape(X.shape[0], -1)

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(
    X_flat, y, test_size=TEST_SPLIT, random_state=42
)

# 3. Scale (CRITICAL for SVM)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Train
model = SVC(kernel=KERNEL_TYPE, C=C_VALUE, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

print(f"\nSVM Accuracy: {acc*100:.2f}%")


# 0.2 RBF = 38.50, 0.2 POLY = 26.32, 0.2 LINEAR = 33.80
# 0.3 RBF = 36.90, 0.3 POLY = 24.54, 0.3 LINEAR = 34.50
