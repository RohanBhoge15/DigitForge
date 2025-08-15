import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os

class NeuralNetwork:
    def __init__(self, input_size=784, hidden_size=64, output_size=10, learning_rate=0.01, momentum=0.9):
        """
        Initialize neural network with 784->64->10 architecture
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        
        # Initialize weights and biases with Xavier initialization
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(2.0 / self.input_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(2.0 / self.hidden_size)
        self.b2 = np.zeros((1, self.output_size))
        
        # Initialize momentum terms
        self.vW1 = np.zeros_like(self.W1)
        self.vb1 = np.zeros_like(self.b1)
        self.vW2 = np.zeros_like(self.W2)
        self.vb2 = np.zeros_like(self.b2)
        
        # Training history
        self.loss_history = []
        self.accuracy_history = []
    
    def sigmoid(self, x):
        """Sigmoid activation function"""
        # Clip x to prevent overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid function"""
        return x * (1 - x)
    
    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """Derivative of ReLU function"""
        return (x > 0).astype(float)
    
    def softmax(self, x):
        """Softmax activation function for output layer"""
        # Subtract max for numerical stability
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward_pass(self, X):
        """
        Forward propagation through the network
        """
        # Input to hidden layer
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)  # Using ReLU for hidden layer
        
        # Hidden to output layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)  # Using softmax for output layer
        
        return self.a2
    
    def compute_loss(self, y_true, y_pred):
        """
        Compute cross-entropy loss
        """
        # Add small epsilon to prevent log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Cross-entropy loss
        loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        return loss
    
    def backward_pass(self, X, y_true, y_pred):
        """
        Backward propagation to compute gradients
        """
        m = X.shape[0]  # Number of samples
        
        # Output layer gradients
        dz2 = y_pred - y_true
        dW2 = (1/m) * np.dot(self.a1.T, dz2)
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        
        # Hidden layer gradients
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.relu_derivative(self.a1)
        dW1 = (1/m) * np.dot(X.T, dz1)
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
        
        return dW1, db1, dW2, db2
    
    def optimize(self, dW1, db1, dW2, db2, use_momentum=True):
        """
        Update weights using SGD with optional momentum
        """
        if use_momentum:
            # Momentum update
            self.vW1 = self.momentum * self.vW1 - self.learning_rate * dW1
            self.vb1 = self.momentum * self.vb1 - self.learning_rate * db1
            self.vW2 = self.momentum * self.vW2 - self.learning_rate * dW2
            self.vb2 = self.momentum * self.vb2 - self.learning_rate * db2
            
            self.W1 += self.vW1
            self.b1 += self.vb1
            self.W2 += self.vW2
            self.b2 += self.vb2
        else:
            # Standard SGD
            self.W1 -= self.learning_rate * dW1
            self.b1 -= self.learning_rate * db1
            self.W2 -= self.learning_rate * dW2
            self.b2 -= self.learning_rate * db2
    
    def predict(self, X):
        """
        Make predictions on new data
        """
        output = self.forward_pass(X)
        return np.argmax(output, axis=1)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities
        """
        return self.forward_pass(X)
    
    def evaluate(self, X, y):
        """
        Evaluate model accuracy
        """
        predictions = self.predict(X)
        y_labels = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == y_labels)
        return accuracy
    
    def save_model(self, filepath):
        """
        Save model parameters to file
        """
        model_data = {
            'W1': self.W1,
            'b1': self.b1,
            'W2': self.W2,
            'b2': self.b2,
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'learning_rate': self.learning_rate,
            'momentum': self.momentum,
            'loss_history': self.loss_history,
            'accuracy_history': self.accuracy_history
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load model parameters from file
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.W1 = model_data['W1']
        self.b1 = model_data['b1']
        self.W2 = model_data['W2']
        self.b2 = model_data['b2']
        self.input_size = model_data['input_size']
        self.hidden_size = model_data['hidden_size']
        self.output_size = model_data['output_size']
        self.learning_rate = model_data['learning_rate']
        self.momentum = model_data['momentum']
        self.loss_history = model_data.get('loss_history', [])
        self.accuracy_history = model_data.get('accuracy_history', [])
        
        print(f"Model loaded from {filepath}")

def one_hot_encode(y, num_classes=10):
    """
    Convert labels to one-hot encoding
    """
    encoded = np.zeros((len(y), num_classes))
    encoded[np.arange(len(y)), y] = 1
    return encoded

def load_mnist_data():
    """
    Load and preprocess MNIST dataset
    """
    print("Loading MNIST dataset...")

    try:
        # Try to load MNIST data from sklearn (OpenML)
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        X, y = mnist.data, mnist.target.astype(int)
        print("✅ MNIST loaded from OpenML")
    except Exception as e:
        print(f"⚠️  OpenML failed ({e}), trying alternative method...")
        try:
            # Alternative: Use sklearn's built-in digits dataset (smaller but works)
            from sklearn.datasets import load_digits
            digits = load_digits()
            X, y = digits.data, digits.target
            print("✅ Using sklearn digits dataset (8x8 images, 1797 samples)")
            print("⚠️  Note: This is a smaller dataset than full MNIST")
        except Exception as e2:
            print(f"❌ All methods failed: {e2}")
            # Create synthetic data as last resort
            print("🔧 Creating synthetic data for testing...")
            np.random.seed(42)
            X = np.random.rand(1000, 784)  # 1000 samples, 784 features
            y = np.random.randint(0, 10, 1000)  # Random labels 0-9
            print("✅ Synthetic data created (1000 samples)")

    # Ensure X has the right shape for our network (784 features)
    if X.shape[1] != 784:
        print(f"⚠️  Reshaping data from {X.shape[1]} to 784 features...")
        if X.shape[1] == 64:  # sklearn digits dataset
            # Pad with zeros to make it 784 features
            padding = np.zeros((X.shape[0], 784 - 64))
            X = np.concatenate([X, padding], axis=1)
        else:
            # Resize to 784 features
            X = np.resize(X, (X.shape[0], 784))

    # Normalize pixel values to [0, 1]
    X = X / np.max(X) if np.max(X) > 1 else X

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Convert labels to one-hot encoding
    y_train_encoded = one_hot_encode(y_train)
    y_test_encoded = one_hot_encode(y_test)

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Input features: {X_train.shape[1]}")

    return X_train, X_test, y_train_encoded, y_test_encoded, y_train, y_test
