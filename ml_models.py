#!/usr/bin/env python3
"""
Professional ML Models for MNIST Digit Recognition
Implements Random Forest, SVM, Gradient Boosting, KNN, and Logistic Regression
"""

import numpy as np
import pickle
import time
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class MLModelComparison:
    def __init__(self):
        """Initialize ML models with optimized parameters for MNIST"""
        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                random_state=42,
                n_jobs=-1
            ),
            'SVM (RBF)': SVC(
                kernel='rbf',
                C=10,
                gamma='scale',
                probability=True,
                random_state=42
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'K-Nearest Neighbors': KNeighborsClassifier(
                n_neighbors=5,
                weights='distance',
                n_jobs=-1
            ),
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                solver='lbfgs',
                random_state=42,
                n_jobs=-1
            )
        }
        
        self.trained_models = {}
        self.training_times = {}
        self.accuracies = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self, sample_size=10000):
        """Load and preprocess MNIST data"""
        print("Loading MNIST dataset...")
        
        # Load MNIST data
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        X, y = mnist.data, mnist.target.astype(int)
        
        # Sample data for faster training (optional)
        if sample_size and sample_size < len(X):
            indices = np.random.choice(len(X), sample_size, replace=False)
            X, y = X[indices], y[indices]
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Normalize data (0-1 range)
        self.X_train = self.X_train / 255.0
        self.X_test = self.X_test / 255.0
        
        print(f"Training set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")

    def train_all_models(self):
        """Train all ML models and record performance"""
        print("\n" + "="*60)
        print("TRAINING PROFESSIONAL ML MODELS")
        print("="*60)
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            start_time = time.time()
            model.fit(self.X_train, self.y_train)
            training_time = time.time() - start_time
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            
            # Store results
            self.trained_models[name] = model
            self.training_times[name] = training_time
            self.accuracies[name] = accuracy
            
            print(f"✅ {name}: {accuracy:.4f} accuracy in {training_time:.2f}s")

    def predict_single(self, image_data, model_name='Random Forest'):
        """Make prediction with specified model"""
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet!")
        
        model = self.trained_models[model_name]
        
        # Ensure image is in correct format
        if image_data.shape != (1, 784):
            image_data = image_data.reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(image_data)[0]
        probabilities = model.predict_proba(image_data)[0]
        confidence = np.max(probabilities)
        
        return {
            'prediction': int(prediction),
            'confidence': float(confidence),
            'probabilities': probabilities.tolist(),
            'model_name': model_name
        }

    def compare_all_models(self, image_data):
        """Compare predictions from all models"""
        results = {}
        
        for model_name in self.trained_models.keys():
            try:
                result = self.predict_single(image_data, model_name)
                results[model_name] = result
            except Exception as e:
                print(f"Error with {model_name}: {e}")
                results[model_name] = {
                    'prediction': -1,
                    'confidence': 0.0,
                    'probabilities': [0.0] * 10,
                    'model_name': model_name,
                    'error': str(e)
                }
        
        return results

    def save_models(self, filename='ml_models.pkl'):
        """Save all trained models"""
        model_data = {
            'models': self.trained_models,
            'training_times': self.training_times,
            'accuracies': self.accuracies
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\n✅ All models saved to {filename}")

    def load_models(self, filename='ml_models.pkl'):
        """Load pre-trained models"""
        try:
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)
            
            self.trained_models = model_data['models']
            self.training_times = model_data['training_times']
            self.accuracies = model_data['accuracies']
            
            print(f"✅ Models loaded from {filename}")
            return True
        except FileNotFoundError:
            print(f"❌ Model file {filename} not found")
            return False

    def print_summary(self):
        """Print performance summary"""
        print("\n" + "="*60)
        print("ML MODELS PERFORMANCE SUMMARY")
        print("="*60)
        
        for name in self.trained_models.keys():
            print(f"{name:20} | Accuracy: {self.accuracies[name]:.4f} | Time: {self.training_times[name]:.2f}s")

def main():
    """Train all ML models"""
    ml_comparison = MLModelComparison()
    
    # Load data
    ml_comparison.load_data(sample_size=20000)  # Use 20k samples for faster training
    
    # Train all models
    ml_comparison.train_all_models()
    
    # Print summary
    ml_comparison.print_summary()
    
    # Save models
    ml_comparison.save_models()
    
    print("\n🎉 All ML models trained and saved successfully!")
    print("You can now use them in the web application for comparison.")

if __name__ == "__main__":
    main()
