import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


INIT_LR    = 0.0001    
BATCH_SIZE = 32        
EPOCHS     = 50         


# 1. LOAD DATA
print("Loading dataset...")
# CHANGE THIS: Load the AUGMENTED data
print("Loading AUGMENTED dataset...")
X = np.load("X_data.npy")  
y = np.load("y_data.npy")  
classes = np.load("classes.npy")

# 2. RESHAPE FOR CNN
# Input Shape: (Samples, 130 Time Steps, 33 Features, 1 Channel)
X = X.reshape(X.shape[0], 130, 33, 1)

# 3. ENCODE & SPLIT
y_onehot = to_categorical(y, num_classes=len(classes))
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

# 4. BUILD CNN MODEL
model = Sequential([
    # Layer 1
    Conv2D(32, (3, 3), activation='relu', input_shape=(130, 33, 1)),
    MaxPooling2D((2, 2)),
    BatchNormalization(),

    # Layer 2
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.3), 

    # Layer 3
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.4),

    # Output
    Flatten(),
    Dense(64, activation='relu'),
    Dense(len(classes), activation='softmax')
])

# 5. COMPILE
opt = tf.keras.optimizers.Adam(learning_rate=INIT_LR)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# 6. SETUP SAVING (Saves the best model automatically)
checkpoint = ModelCheckpoint(
    "best_cnn_model.h5", 
    monitor='val_accuracy', 
    verbose=1, 
    save_best_only=True, 
    mode='max'
)

# 7. TRAIN
print("\n--- STARTING TRAINING ---")
history = model.fit(
    X_train, y_train, 
    epochs=EPOCHS, 
    batch_size=BATCH_SIZE, 
    validation_data=(X_test, y_test),
    callbacks=[checkpoint]
)

# 8. LOAD BEST & SHOW RESULT
print("\nLoading the best model saved during training...")
best_model = tf.keras.models.load_model("best_cnn_model.h5")
loss, acc = best_model.evaluate(X_test, y_test)
print(f"\n🚀 FINAL ACCURACY: {acc*100:.2f}%")

# 9. PLOT
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Test Acc')
plt.title(f"CNN Training Progress (Final: {acc*100:.1f}%)")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
