import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


TEST_SPLIT = 0.2    
MAX_DEPTH = 20       


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

print(f"\nDecision Tree Accuracy: {acc*100:.2f}%")
print(f"   (Max Depth: {MAX_DEPTH})")
print("\nDetailed Report:")
print(classification_report(y_test, preds, target_names=classes))

# 0.2 No Depth = 24.38%, 0.2 10(Depth) = 23.27%, 0.2 20(Depth) = 27.15%, 0.2 30(Depth) = 24.38%
# 0.3 No Depth = 21.40%, 0.3 10 (Depth) = 23.60%,0.3 20(Depth) = 21.40%, 0.3 30(Depth) = 21.40%