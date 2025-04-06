# Fashion MNIST Image Classification

A comprehensive machine learning project implementing various models for classifying Fashion MNIST images.

## 📋 Overview

This project explores different machine learning approaches to classify clothing items from the Fashion MNIST dataset. The repository contains various model implementations (CNN, Logistic Regression) and utilities for data processing and evaluation.

## 🔍 The Dataset

The Fashion MNIST dataset consists of 60,000 training images and 10,000 test images of clothing items across 10 categories:
- T-shirt/top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot

Each image is a 28x28 grayscale image, associated with a label from 10 classes. The project uses the TensorFlow implementation of the Fashion MNIST dataset.

## 🚀 Project Structure

```
fashion-mnist-classification/
├── src/                         # Source code
│   ├── models/                  # Model implementations
│   │   ├── __init__.py          # Package initialization
│   │   ├── lr.py                # Logistic Regression model
│   │   ├── cnn1.py              # CNN model implementation 1
│   │   ├── cnn2.py              # CNN model implementation 2
│   ├── utils/                   # Utility functions
│   │   ├── __init__.py          # Package initialization
│   │   ├── ubyte.py             # Utilities for loading and visualizing Fashion MNIST
│   ├── __init__.py              # Package initialization
│   ├── main.py                  # Main execution script
│   ├── task0.py                 # Initial data exploration
├── requirements.txt             # Project dependencies
├── .gitignore                   # Git ignore file
└── README.md                    # Project documentation
```

## 🔧 Implementation Details

### Models

1. **Logistic Regression (`src/models/lr.py`)**
   - Basic machine learning model for classification
   - Uses scikit-learn's LogisticRegression with the saga solver

2. **Convolutional Neural Network 1 (`src/models/cnn1.py`)**
   - CNN architecture with convolutional layers, max pooling, and dense layers
   - Uses LeakyReLU activation
   - Simple architecture with two convolutional layers

3. **Convolutional Neural Network 2 (`src/models/cnn2.py`)**
   - Enhanced CNN architecture with three convolutional layers
   - Features batch normalization and dropout for regularization
   - Deeper network with additional dense layers

### Utilities

- **Data Loading (`src/utils/ubyte.py`)**
  - Functions to load Fashion MNIST dataset from TensorFlow
  - Data visualization utilities
  - Functions to save processed data to CSV

- **Data Exploration (`src/task0.py`)**
  - Initial dataset loading and visualization
  - Basic statistics and data insights
  - Pixel value distribution analysis

- **Model Comparison (`src/main.py`)**
  - Unified script to run and compare all models
  - Generates performance visualization
  - Standardized data preprocessing

## 📊 Results

The different models achieve varying accuracy on the Fashion MNIST dataset:

| Model | Accuracy |
|-------|----------|
| Logistic Regression | ~85% |
| CNN Model 1 | ~91% |
| CNN Model 2 | ~93% |

(Note: Actual accuracies may vary based on hyperparameters and training conditions)

## 🚀 Getting Started

### Prerequisites

- Python 3.6+
- TensorFlow 2.x
- NumPy
- Pandas
- Matplotlib
- scikit-learn

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/fashion-mnist-classification.git

# Navigate to the project directory
cd fashion-mnist-classification

# Install dependencies
pip install -r requirements.txt
```

### Running the Models

To run all models and compare their performance:

```bash
# Run the main script
python src/main.py
```

To run individual models:

```bash
# Run CNN model 1
python src/models/cnn1.py

# Run logistic regression model
python src/models/lr.py

# Run data exploration
python src/task0.py
```

## 📝 Future Improvements

- Implement model improvement strategies like data augmentation
- Explore transfer learning approaches
- Add hyperparameter tuning
- Implement additional architectures (ResNet, EfficientNet, etc.)
- Create a web interface for model demonstration

## 📚 References

- [Fashion MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)
- [Convolutional Neural Networks](https://www.tensorflow.org/tutorials/images/cnn)
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs) 