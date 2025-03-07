import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input,Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LeakyReLU
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical

# Load dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train, y_train = x_train[:50000], y_train[:50000]
x_test, y_test = x_test[:10000], y_test[:10000]

# Normalize and reshape images
x_train = x_train / 255.0
x_test = x_test / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# One-hot encode
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Define CNN model
model = Sequential([
    Input(shape=(28,28,1)),
    Conv2D(32, (3,3)),
    LeakyReLU(negative_slope=0.1),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3)),
    MaxPooling2D((2,2)),
    Flatten(),
    LeakyReLU(negative_slope=0.1),
    Dense(10, activation='softmax')  # Output layer
])

# Compile and train the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"CNN Model Accuracy: {test_acc:.4f}")
