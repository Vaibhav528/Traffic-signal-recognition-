import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

# Set the path to the dataset directory
dataset_path = os.path.join(os.getcwd(), 'train')

# Retrieve the images and their labels
data = []
labels = []

for class_index in range(43):
    path = os.path.join(dataset_path, str(class_index))
    images = os.listdir(path)

    for image_name in images:
        try:
            image_path = os.path.join(path, image_name)
            image = Image.open(image_path)
            image = image.resize((30, 30))
            image_array = np.array(image)
            data.append(image_array)
            labels.append(class_index)
        except:
            print("Error loading image:", image_path)

# Convert lists into numpy arrays
data = np.array(data)
labels = np.array(labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Convert the labels into one-hot encoding
y_train = to_categorical(y_train, num_classes=43)
y_test = to_categorical(y_test, num_classes=43)

# Build the model
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=(30, 30, 3)))
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(43, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
epochs = 15
history = model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test))
model.save("my_model.h5")

# Plotting graphs for accuracy and loss
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Load the test dataset
test_data = pd.read_csv('Test.csv')
test_labels = test_data["ClassId"].values
test_images = test_data["Path"].values

# Test the model on the test dataset
test_images_data = []

for image_path in test_images:
    try:
        image = Image.open(image_path)
        image = image.resize((30, 30))
        image_array = np.array(image)
        test_images_data.append(image_array)
    except:
        print("Error loading image:", image_path)

test_images_data = np.array(test_images_data)

predictions = model.predict_classes(test_images_data)

# Calculate accuracy on the test dataset
accuracy = accuracy_score(test_labels, predictions)
print("Test Accuracy:", accuracy)
