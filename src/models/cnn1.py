"""
CNN Model 1 for Fashion MNIST classification
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LeakyReLU
from tensorflow.keras.utils import to_categorical

def create_cnn1_model():
    """
    Create a CNN model with the architecture:
    - Conv2D(32, 3x3) with LeakyReLU
    - MaxPooling2D(2x2)
    - Conv2D(64, 3x3)
    - MaxPooling2D(2x2)
    - Flatten
    - Dense(10) with softmax
    
    Returns:
        tf.keras.Model: Compiled CNN model
    """
    model = Sequential([
        Input(shape=(28, 28, 1)),
        Conv2D(32, (3, 3)),
        LeakyReLU(negative_slope=0.1),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3)),
        MaxPooling2D((2, 2)),
        Flatten(),
        LeakyReLU(negative_slope=0.1),
        Dense(10, activation='softmax')  # Output layer
    ])
    
    model.compile(optimizer='adam', 
                 loss='categorical_crossentropy', 
                 metrics=['accuracy'])
    
    return model

def train_cnn1(x_train, y_train, x_test, y_test, epochs=10, batch_size=32):
    """
    Train CNN Model 1 on the Fashion MNIST dataset
    
    Args:
        x_train: Training images (N, 28, 28, 1)
        y_train: Training labels
        x_test: Test images (N, 28, 28, 1)
        y_test: Test labels
        epochs: Number of training epochs
        batch_size: Batch size for training
        
    Returns:
        float: Accuracy of the model on test data
    """
    print("Training CNN Model 1...")
    
    # One-hot encode the labels
    y_train_one_hot = to_categorical(y_train, 10)
    y_test_one_hot = to_categorical(y_test, 10)
    
    # Create and train model
    model = create_cnn1_model()
    
    # Train the model
    model.fit(x_train, y_train_one_hot, 
              epochs=epochs, 
              batch_size=batch_size, 
              validation_data=(x_test, y_test_one_hot),
              verbose=1)
    
    # Evaluate the model
    _, accuracy = model.evaluate(x_test, y_test_one_hot, verbose=0)
    print(f"CNN Model 1 Accuracy: {accuracy:.4f}")
    
    return accuracy

if __name__ == "__main__":
    # For standalone execution
    from tensorflow.keras.datasets import fashion_mnist
    
    # Load data
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    
    # Normalize and reshape
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    
    # Train and evaluate
    train_cnn1(x_train, y_train, x_test, y_test)
