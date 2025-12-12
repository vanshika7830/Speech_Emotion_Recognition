import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

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
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\nDecision Tree Accuracy: {acc*100:.2f}%")
print(f"   (Max Depth: {MAX_DEPTH})")
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Visualization
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=classes, yticklabels=classes)
plt.title(f"Confusion Matrix (Acc: {acc*100:.1f}%)")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Plot 2: Feature Importance
plt.figure(figsize=(8,6))
importances = model.feature_importances_
imp_reshaped = importances.reshape(130, 33).mean(axis=0)
indices = np.argsort(imp_reshaped)[::-1][:10]

feat_names = ([f"MFCC_{i+1}" for i in range(13)] + 
              [f"Chroma_{i+1}" for i in range(12)] + 
              [f"Contrast_{i+1}" for i in range(7)] + 
              ["Bandwidth"])
top_names = [feat_names[i] for i in indices]

plt.bar(range(10), imp_reshaped[indices], color='orange')
plt.xticks(range(10), top_names, rotation=45, ha='right')
plt.title("Top 10 Features (Decision Tree)")

plt.tight_layout()
plt.show()
# 0.2 No Depth = 24.38%, 0.2 10(Depth) = 23.27%, 0.2 20(Depth) = 27.15%, 0.2 30(Depth) = 24.38%
# 0.3 No Depth = 21.40%, 0.3 10 (Depth) = 23.60%,0.3 20(Depth) = 21.40%, 0.3 30(Depth) = 21.40%