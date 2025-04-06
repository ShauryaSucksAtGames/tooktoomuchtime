"""
Logistic Regression model for Fashion MNIST classification
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical

def train_logistic_regression(x_train, y_train, x_test, y_test):
    """
    Train a logistic regression model on the Fashion MNIST dataset
    
    Args:
        x_train: Training images
        y_train: Training labels
        x_test: Test images
        y_test: Test labels
        
    Returns:
        float: Accuracy of the model on test data
    """
    print("Training Logistic Regression model...")
    
    # Flatten images for logistic regression
    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    x_test_flat = x_test.reshape(x_test.shape[0], -1)
    
    # Standardize features
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_flat)
    x_test_scaled = scaler.transform(x_test_flat)
    
    # Create and train model
    model = LogisticRegression(max_iter=300, solver='saga', n_jobs=-1)
    model.fit(x_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(x_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Logistic Regression Accuracy: {accuracy:.4f}")
    
    return accuracy

if __name__ == "__main__":
    # For standalone execution
    from tensorflow.keras.datasets import fashion_mnist
    
    # Load data
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    
    # Normalize
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    
    # Train and evaluate
    train_logistic_regression(x_train, y_train, x_test, y_test)