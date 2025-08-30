import os
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Load and preprocess dataset (UTKFace example)
dataset_path = 'path_to_UTKFace_dataset'
images = []
labels = []

for filename in os.listdir(dataset_path):
    image_path = os.path.join(dataset_path, filename)
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  # Resize to 224x224
    images.append(img)

    # Gender: 0 for male, 1 for female (in UTKFace dataset, gender is stored as the second digit in the filename)
    label = int(filename.split('_')[1])  # Extract gender label from filename
    labels.append(label)

images = np.array(images, dtype='float32')
labels = np.array(labels)

# Normalize image data
images = images / 255.0

# Convert labels to one-hot encoding
labels = to_categorical(labels, 2)  # Two classes: male (0) and female (1)

# Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')  # Two output classes: male and female
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

# Save the model
model.save('gender_model.h5')
