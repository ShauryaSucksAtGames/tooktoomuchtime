"""
Fashion MNIST Classification - Main execution script
This script loads the Fashion MNIST dataset and runs all implemented models for comparison.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist

# Ensure src is in path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_data():
    """Load and preprocess the Fashion MNIST dataset"""
    print("Loading Fashion MNIST dataset...")
    
    # Load data
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    
    # Normalize and reshape
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    
    print(f"Training data shape: {x_train.shape}")
    print(f"Test data shape: {x_test.shape}")
    
    return (x_train, y_train), (x_test, y_test)

def run_all_models():
    """Run all available models and compare results"""
    (x_train, y_train), (x_test, y_test) = load_data()
    
    # Import model functions
    from src.models.lr import train_logistic_regression
    from src.models.cnn1 import train_cnn1
    from src.models.cnn2 import train_cnn2
    
    # Dictionary to store results
    results = {}
    
    # Run Logistic Regression
    print("\n=== Running Logistic Regression ===")
    lr_acc = train_logistic_regression(x_train, y_train, x_test, y_test)
    results["Logistic Regression"] = lr_acc
    
    # Reshape for CNN models
    x_train_cnn = x_train.reshape(-1, 28, 28, 1)
    x_test_cnn = x_test.reshape(-1, 28, 28, 1)
    
    # Run CNN1
    print("\n=== Running CNN Model 1 ===")
    cnn1_acc = train_cnn1(x_train_cnn, y_train, x_test_cnn, y_test)
    results["CNN Model 1"] = cnn1_acc
    
    # Run CNN2
    print("\n=== Running CNN Model 2 ===")
    cnn2_acc = train_cnn2(x_train_cnn, y_train, x_test_cnn, y_test)
    results["CNN Model 2"] = cnn2_acc
    
    # Display comparison results
    print("\n=== Model Comparison ===")
    for model, accuracy in results.items():
        print(f"{model}: {accuracy:.4f} accuracy")
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.bar(results.keys(), results.values())
    plt.title("Model Accuracy Comparison")
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    for i, v in enumerate(results.values()):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
    plt.savefig("model_comparison.png")
    plt.show()
    
if __name__ == "__main__":
    run_all_models() 