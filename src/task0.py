"""
Initial data exploration for Fashion MNIST dataset
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.ubyte import visualize_samples

def explore_fashion_mnist():
    """
    Perform initial exploration of the Fashion MNIST dataset
    """
    print("Exploring Fashion MNIST dataset...")
    
    # Load data using TensorFlow
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    
    # Print dataset shapes
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Create DataFrame from a sample of the data
    df_sample = pd.DataFrame(X_train[:1000].reshape(1000, -1))
    df_sample['label'] = y_train[:1000]
    
    # Print dataframe info
    print("\nDataFrame Sample Info:")
    print(df_sample.info())
    print("\nLabel distribution:")
    print(df_sample['label'].value_counts().sort_index())
    
    # Define class names
    class_names = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", 
                  "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    
    # Display sample images
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        # Show one image from each class
        class_idx = i
        img_idx = np.where(y_train == class_idx)[0][0]
        ax.imshow(X_train[img_idx], cmap='gray')
        ax.set_title(f"{class_idx}: {class_names[class_idx]}")
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig("fashion_mnist_classes.png")
    plt.show()
    
    # Analyze pixel values
    sample_image = X_train[0]
    print(f"\nSample image shape: {sample_image.shape}")  # Should be (28, 28)
    print(f"Pixel value range: {X_train.min()} to {X_train.max()}")  # 0 to 255
    
    # Plot pixel value distribution
    plt.figure(figsize=(10, 6))
    plt.hist(X_train.flatten(), bins=50)
    plt.title('Pixel Value Distribution')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.savefig("pixel_distribution.png")
    plt.show()
    
    return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = explore_fashion_mnist() 