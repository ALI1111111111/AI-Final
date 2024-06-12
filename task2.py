import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load CIFAR-10 dataset using TensorFlow
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Preprocess the data
X_train = np.float32(X_train) / 255.0
X_test = np.float32(X_test) / 255.0
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

# Define the CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.1)

# Evaluate the model
_, accuracy = model.evaluate(X_test, y_test)
print("Accuracy:", accuracy)


import matplotlib.pyplot as plt

# Make predictions on the test set
predictions = model.predict(X_test)

# Visualize some sample predictions
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.imshow(X_test[i])
    predicted_label = np.argmax(predictions[i])
    true_label = np.argmax(y_test[i])
    plt.title(f'Predicted: {predicted_label}, True: {true_label}')
    plt.axis('off')
plt.show()

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load your own image
image_path = "bird.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
resized_image = cv2.resize(image, (32, 32))  # Resize to match model input size

# Preprocess the image
preprocessed_image = resized_image.astype(np.float32) / 255.0  # Normalize pixel values
preprocessed_image = np.expand_dims(preprocessed_image, axis=0)  # Add batch dimension

# Make prediction using the model
prediction = model.predict(preprocessed_image)
predicted_class = np.argmax(prediction)

# Load CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Display the image and predicted class
plt.imshow(resized_image)
plt.title(f"Predicted class: {class_names[predicted_class]}")
plt.axis('off')
plt.show()