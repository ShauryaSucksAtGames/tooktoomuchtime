import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

df_train = pd.DataFrame(X_train.reshape(X_train.shape[0], -1)) 
df_train['label'] = y_train

# Print dataset shape
print(f"Training data shape: {X_train.shape}")  
print(f"Test data shape: {X_test.shape}")       

# Define class names
class_names = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", 
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Display a few images
fig, axes = plt.subplots(1, 10, figsize=(10, 4))
for i, ax in enumerate(axes):
    ax.imshow(X_train[i], cmap='gray')  # Show image
    ax.set_title(class_names[y_train[i]])  # Show label
    ax.axis('off')
plt.show()

# Verify grayscale format by checking shape and pixel values
sample_image = X_train[0]
print(f"Sample image shape: {sample_image.shape}")  # Should be (28, 28)
print(f"Pixel value range: {sample_image.min()} to {sample_image.max()}")  # 0 to 255 