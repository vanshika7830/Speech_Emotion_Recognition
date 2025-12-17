import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

TEST_SPLIT = 0.2  
KERNEL_TYPE = 'linear'     
C_VALUE = 10.0           # (Higher = stricter margin)


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

# 3. Scale 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Train
model = SVC(kernel=KERNEL_TYPE, C=C_VALUE, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\nSVM Accuracy: {acc*100:.2f}%")
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Visualization
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title(f"Confusion Matrix (Acc: {acc*100:.1f}%)")
plt.show()


# Plot 2: 2D SCATTER PLOT (Decision Boundary)
# We use PCA to reduce 4290 dimensions -> 2 dimensions just for this plot
print("Calculating 2D projection for scatter plot...")
pca = PCA(n_components=2)
X_vis = pca.fit_transform(X_test) # Squash test data to 2D

# We need to re-train a mini-SVM on just these 2 dimensions to show the boundary lines
svm_2d = SVC(kernel='rbf', C=C_VALUE)
svm_2d.fit(X_vis, y_test)

plt.figure(figsize=(8,6))
# Create a meshgrid to show the decision areas
h = .2  # step size in the mesh
x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Predict on the meshgrid
Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot contours and data points
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
scatter = plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y_test, cmap='coolwarm', edgecolors='k', s=20)
plt.title("SVM Decision Boundary (PCA Projection)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.tight_layout()
plt.show()

# C_Value(10)
# 0.2 RBF  = 42.66%, 0.2 POLY = 26.32%, 0.2 LINEAR = 33.80%
# 0.3 RBF = 36.90%, 0.3 POLY = 24.54%, 0.3 LINEAR = 34.50%


# Scatter plot is showing that emotions are mixed together and hard to separate with a straight line).