import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# Step 1: Load images and labels
def load_data(data_dir, img_size=(128, 128)):
    images = []
    labels = []
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        for image_file in os.listdir(label_dir):
            image_path = os.path.join(label_dir, image_file)
            # Load and resize image
            image = cv2.imread(image_path)
            image = cv2.resize(image, img_size)
            images.append(image)
            labels.append(label)
    return np.array(images), np.array(labels)

data_dir = 'path/to/your/data'
images, labels = load_data(data_dir)

# Step 2: Preprocess data
# Normalize pixel values
images = images / 255.0

# Encode labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
labels = to_categorical(labels)

# Step 3: Split data into training and testing sets
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# Step 4: Define neural network model
input_shape = train_images.shape[1:]  # (image_height, image_width, num_channels)
num_classes = labels.shape[1]  # Number of unique labels

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Step 5: Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Step 6: Train the model
model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# Step 7: Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
