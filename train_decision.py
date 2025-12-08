import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# ================= CONFIGURATION =================
TEST_SPLIT = 0.3        # Trees often need more training data
MAX_DEPTH = None        # Try None (full tree), 10, 20 (pruned tree)
# =================================================

print("--- TRAINING DECISION TREE ---")

# 1. Load & Flatten
X = np.load("X_data.npy")
y = np.load("y_data.npy")
classes = np.load("classes.npy")

X_flat = X.reshape(X.shape[0], -1)

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(
    X_flat, y, test_size=TEST_SPLIT, random_state=42
)

# 3. Train
model = DecisionTreeClassifier(max_depth=MAX_DEPTH, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluate
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

print(f"\n✅ Decision Tree Accuracy: {acc*100:.2f}%")
# print(f"   (Max Depth: {MAX_DEPTH})")
# print("\nDetailed Report:")
# print(classification_report(y_test, preds, target_names=classes))