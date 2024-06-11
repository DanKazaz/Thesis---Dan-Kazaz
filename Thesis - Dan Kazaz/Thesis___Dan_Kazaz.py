import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

# Custom to_categorical function
def to_categorical(y, num_classes):
    return np.eye(num_classes)[y]

# Utility functions
def conv2d(image, kernel):
    kernel_height, kernel_width = kernel.shape
    image_height, image_width = image.shape
    output_height = image_height - kernel_height + 1
    output_width = image_width - kernel_width + 1
    output = np.zeros((output_height, output_width))

    for i in range(output_height):
        for j in range(output_width):
            output[i, j] = np.sum(image[i:i + kernel_height, j:j + kernel_width] * kernel)
    return output

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def max_pooling(image, size=2, stride=2):
    image_height, image_width = image.shape
    output_height = (image_height - size) // stride + 1
    output_width = (image_width - size) // stride + 1
    output = np.zeros((output_height, output_width))

    for i in range(0, image_height - size + 1, stride):
        for j in range(0, image_width - size + 1, stride):
            output[i // stride, j // stride] = np.max(image[i:i + size, j:j + size])
    return output

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def cross_entropy_loss(predictions, targets):
    return -np.sum(targets * np.log(predictions))

def cross_entropy_loss_derivative(predictions, targets):
    return predictions - targets

# Load data function
def load_data(data_dir, img_size=(28, 28)):
    images = []
    labels = []
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            for image_file in os.listdir(label_dir):
                image_path = os.path.join(label_dir, image_file)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    print(f"Warning: Unable to read file {image_path}. Skipping...")
                    continue
                image = cv2.resize(image, img_size)
                images.append(image)
                labels.append(label)
        else:
            print(f"Skipping non-directory {label_dir}")
    return np.array(images), np.array(labels)

# Set the correct path to your dataset
data_dir = r'C:\Users\User\source\repos\Thesis - Dan Kazaz\Thesis - Dan Kazaz\data'  # Update this path

# Load data
images, labels = load_data(data_dir)

# Debugging: Print the number of loaded images and labels
print(f"Loaded {len(images)} images and {len(labels)} labels.")

# Ensure there are images loaded
if len(images) == 0:
    raise ValueError("No images found. Please check the data directory path and ensure it contains image files.")

# Preprocess data
images = images / 255.0
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
labels = to_categorical(labels, num_classes=len(np.unique(labels)))

# Split data
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# CNN layers and model
class ConvLayer:
    def __init__(self, num_kernels, kernel_size, input_shape):
        self.num_kernels = num_kernels
        self.kernel_size = kernel_size
        self.kernels = np.random.randn(num_kernels, *kernel_size) * 0.1
        self.input_shape = input_shape

    def forward(self, X):
        self.input = X
        self.output = np.array([conv2d(X, kernel) for kernel in self.kernels])
        return self.output

    def backward(self, dL_dout, learning_rate):
        dL_dk = np.zeros(self.kernels.shape)
        for i in range(self.num_kernels):
            dL_dk[i] = conv2d(self.input, dL_dout[i])
        self.kernels -= learning_rate * dL_dk

class FullyConnectedLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size) * 0.1
        self.biases = np.zeros((output_size, 1))

    def forward(self, X):
        self.input = X
        self.output = np.dot(self.weights, X) + self.biases
        return self.output

    def backward(self, dL_dout, learning_rate):
        dL_dw = np.dot(dL_dout, self.input.T)
        dL_db = np.sum(dL_dout, axis=1, keepdims=True)
        dL_dx = np.dot(self.weights.T, dL_dout)
        self.weights -= learning_rate * dL_dw
        self.biases -= learning_rate * dL_db
        return dL_dx

class SimpleCNN:
    def __init__(self, input_shape, num_classes):
        self.conv = ConvLayer(8, (3, 3), input_shape)
        self.fc = FullyConnectedLayer(13 * 13 * 8, num_classes)  # Adjust size accordingly
        self.num_classes = num_classes

    def forward(self, X):
        X = self.conv.forward(X)
        X = relu(X)
        X = max_pooling(X)
        X = X.flatten().reshape(-1, 1)
        X = self.fc.forward(X)
        X = softmax(X)
        return X

    def backward(self, X, Y, learning_rate):
        dout = cross_entropy_loss_derivative(X, Y)
        dout = self.fc.backward(dout, learning_rate)
        dout = dout.reshape(8, 13, 13)
        dout = relu_derivative(dout)
        self.conv.backward(dout, learning_rate)

    def train(self, X_train, Y_train, epochs, learning_rate):
        for epoch in range(epochs):
            X_train, Y_train = shuffle(X_train, Y_train)
            for X, Y in zip(X_train, Y_train):
                X = X.reshape(self.conv.input_shape)
                output = self.forward(X)
                self.backward(output, Y, learning_rate)
            print(f'Epoch {epoch + 1}/{epochs} completed')

    def predict(self, X):
        output = self.forward(X.reshape(self.conv.input_shape))
        return np.argmax(output)

    def evaluate(self, X_test, Y_test):
        predictions = [self.predict(X) for X in X_test]
        accuracy = accuracy_score(Y_test, predictions)
        print(f'Accuracy: {accuracy * 100:.2f}%')

# Initialize and train the model
input_shape = train_images[0].shape
num_classes = len(np.unique(labels))
cnn = SimpleCNN(input_shape, num_classes)
cnn.train(train_images, train_labels, epochs=10, learning_rate=0.01)

# Evaluate the model
cnn.evaluate(test_images, np.argmax(test_labels, axis=1))
