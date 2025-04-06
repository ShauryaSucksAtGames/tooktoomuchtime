"""
Utility functions for working with Fashion MNIST dataset from TensorFlow
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def load_fashion_mnist():
    """
    Load the Fashion MNIST dataset from TensorFlow
    
    Returns:
        tuple: (x_train, y_train) containing the images and labels
    """
    # Load data using TensorFlow
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    
    return X_train, y_train

def visualize_samples(X, y, num_samples=7, class_names=None):
    """
    Visualize sample images with their labels
    
    Args:
        X: Array of images
        y: Array of labels
        num_samples: Number of samples to visualize
        class_names: List of class names (if None, will use index as label)
    """
    if class_names is None:
        class_names = [str(i) for i in range(10)]
    
    fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 1.5, 3))
    
    for i, ax in enumerate(axes):
        if i < len(X):
            ax.imshow(X[i], cmap="gray")
            ax.set_title(class_names[y[i]])
            ax.axis("off")
    
    plt.tight_layout()
    return fig

def save_to_csv(X, y, output_file="data_processed.csv"):
    """
    Save image and label data to a CSV file
    
    Args:
        X: Array of images
        y: Array of labels
        output_file: Path to the output CSV file
    """
    # Flatten images and stack with labels
    X_flat = X.reshape(X.shape[0], -1)
    data = np.column_stack((y, X_flat))
    
    # Save to CSV
    header = ["label"] + [f"pixel_{i}" for i in range(X_flat.shape[1])]
    np.savetxt(output_file, data, delimiter=",", header=",".join(header), comments="")
    
    print(f"Data saved to {output_file}")

if __name__ == "__main__":
    # Example usage
    # Define class names
    class_names = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", 
                  "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    
    # Load data
    X, y = load_fashion_mnist()
    
    print(f"Dataset shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    
    # Visualize samples
    visualize_samples(X, y, class_names=class_names)
    plt.show()
    
    # Verify grayscale format
    print(f"Sample image shape: {X[0].shape}")
    print(f"Pixel values range: {X.min()} to {X.max()}")
