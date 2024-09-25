import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Path of data set#

data_dir = 'D:/Data Analyst/Python Models/Defect detection'

categories = ['faulty', 'non_faulty']
data = []
labels = []

# Loading Images #

for category in categories:
    folder_path = os.path.join(data_dir, category)
    label = categories.index(category)  # 0 for non-faulty, 1 for faulty
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (128, 128))  # Resize images to 128x128
        data.append(img)
        labels.append(label)


# Convert to numpy arrays #

data = np.array(data).reshape(-1, 128, 128, 1)  # Reshape for CNN
labels = np.array(labels)

# Normalize the data

data = data / 255.0

# Spliting into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Build the model
model = Sequential()

# Add Convolutional Layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())

# Add Dense Layers (Fully connected layers)
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # Dropout to avoid overfitting

model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')


model.save('image_detection.h5')

