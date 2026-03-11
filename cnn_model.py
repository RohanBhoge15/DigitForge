"""
CNN Model for MNIST Digit Recognition
Built using TensorFlow/Keras for comparison with custom NumPy neural network
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import pickle
import time
import os


class CNNModel:
    def __init__(self):
        """Initialize CNN model for MNIST digit recognition"""
        self.model = None
        self.history = None
        self.training_time = 0
        self.test_accuracy = 0
        self.model_path = 'cnn_model.h5'
        self.metrics_path = 'cnn_metrics.pkl'
        
    def build_model(self):
        """
        Build CNN architecture optimized for MNIST
        Architecture:
        - Conv2D (32 filters, 3x3) + ReLU + MaxPooling
        - Conv2D (64 filters, 3x3) + ReLU + MaxPooling
        - Flatten
        - Dense (128) + ReLU + Dropout
        - Dense (10) + Softmax
        """
        model = models.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), 
                         padding='same', name='conv1'),
            layers.MaxPooling2D((2, 2), name='pool1'),
            layers.BatchNormalization(name='bn1'),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2'),
            layers.MaxPooling2D((2, 2), name='pool2'),
            layers.BatchNormalization(name='bn2'),
            
            # Third Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv3'),
            layers.BatchNormalization(name='bn3'),
            
            # Flatten and Dense Layers
            layers.Flatten(name='flatten'),
            layers.Dense(128, activation='relu', name='dense1'),
            layers.Dropout(0.5, name='dropout'),
            layers.Dense(10, activation='softmax', name='output')
        ])
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def load_data(self):
        """Load and preprocess MNIST dataset"""
        print("Loading MNIST dataset...")
        
        # Load MNIST data
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        
        # Reshape for CNN (add channel dimension)
        X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        
        # One-hot encode labels
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)
        
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Test samples: {X_test.shape[0]}")
        
        return (X_train, y_train), (X_test, y_test)
    
    def train(self, epochs=10, batch_size=128, validation_split=0.1):
        """Train the CNN model"""
        print("\n🚀 Training CNN Model...")
        print("="*60)
        
        # Load data
        (X_train, y_train), (X_test, y_test) = self.load_data()
        
        # Build model if not already built
        if self.model is None:
            self.build_model()
        
        # Print model summary
        print("\n📊 CNN Architecture:")
        self.model.summary()
        
        # Callbacks for better training
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=1e-6
            )
        ]
        
        # Train model
        print(f"\n🏋️ Training for {epochs} epochs...")
        start_time = time.time()
        
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        self.training_time = time.time() - start_time
        
        # Evaluate on test set
        print("\n📈 Evaluating on test set...")
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        self.test_accuracy = test_accuracy
        
        print(f"\n✅ Training completed in {self.training_time:.2f}s")
        print(f"📊 Test Accuracy: {test_accuracy*100:.2f}%")
        print(f"📉 Test Loss: {test_loss:.4f}")
        
        return self.history
    
    def save_model(self):
        """Save trained model and metrics"""
        # Save Keras model
        self.model.save(self.model_path)
        print(f"✅ CNN model saved to {self.model_path}")
        
        # Save metrics
        metrics = {
            'training_time': self.training_time,
            'test_accuracy': self.test_accuracy,
            'history': {
                'loss': self.history.history['loss'],
                'accuracy': self.history.history['accuracy'],
                'val_loss': self.history.history['val_loss'],
                'val_accuracy': self.history.history['val_accuracy']
            }
        }
        
        with open(self.metrics_path, 'wb') as f:
            pickle.dump(metrics, f)
        print(f"✅ CNN metrics saved to {self.metrics_path}")
    
    def load_model(self):
        """Load trained model and metrics"""
        if os.path.exists(self.model_path):
            self.model = keras.models.load_model(self.model_path)
            print(f"✅ CNN model loaded from {self.model_path}")
            
            # Load metrics if available
            if os.path.exists(self.metrics_path):
                with open(self.metrics_path, 'rb') as f:
                    metrics = pickle.load(f)
                self.training_time = metrics.get('training_time', 0)
                self.test_accuracy = metrics.get('test_accuracy', 0)
                print(f"✅ CNN metrics loaded (Accuracy: {self.test_accuracy*100:.2f}%)")
            
            return True
        else:
            print(f"❌ No trained CNN model found at {self.model_path}")
            return False
    
    def predict(self, image):
        """
        Make prediction on a single image
        Args:
            image: numpy array of shape (1, 784) or (28, 28) or (1, 28, 28, 1)
        Returns:
            predicted class (0-9)
        """
        # Reshape input to (1, 28, 28, 1) if needed
        if image.shape == (1, 784):
            image = image.reshape(1, 28, 28, 1)
        elif image.shape == (28, 28):
            image = image.reshape(1, 28, 28, 1)
        elif image.shape == (784,):
            image = image.reshape(1, 28, 28, 1)
        
        # Make prediction
        predictions = self.model.predict(image, verbose=0)
        return np.argmax(predictions, axis=1)[0]
    
    def predict_proba(self, image):
        """
        Get probability distribution for all classes
        Args:
            image: numpy array of shape (1, 784) or (28, 28) or (1, 28, 28, 1)
        Returns:
            probability distribution array of shape (10,)
        """
        # Reshape input to (1, 28, 28, 1) if needed
        if image.shape == (1, 784):
            image = image.reshape(1, 28, 28, 1)
        elif image.shape == (28, 28):
            image = image.reshape(1, 28, 28, 1)
        elif image.shape == (784,):
            image = image.reshape(1, 28, 28, 1)
        
        # Get probabilities
        predictions = self.model.predict(image, verbose=0)
        return predictions[0]
    
    def get_model_info(self):
        """Get model architecture information"""
        if self.model is None:
            return None
        
        total_params = self.model.count_params()
        trainable_params = sum([tf.size(w).numpy() for w in self.model.trainable_weights])
        
        return {
            'architecture': 'CNN (Conv2D-MaxPool-Conv2D-MaxPool-Conv2D-Dense)',
            'total_parameters': int(total_params),
            'trainable_parameters': int(trainable_params),
            'input_shape': '(28, 28, 1)',
            'output_classes': 10,
            'optimizer': 'Adam',
            'loss_function': 'Categorical Crossentropy'
        }


def train_cnn_model(epochs=10, batch_size=128):
    """Standalone function to train CNN model"""
    cnn = CNNModel()
    cnn.build_model()
    cnn.train(epochs=epochs, batch_size=batch_size)
    cnn.save_model()
    return cnn


if __name__ == "__main__":
    # Train CNN model
    print("🧠 CNN Model Training Script")
    print("="*60)
    
    cnn = train_cnn_model(epochs=15, batch_size=128)
    
    print("\n🎉 CNN training complete!")
    print(f"Model saved to: {cnn.model_path}")
    print(f"Metrics saved to: {cnn.metrics_path}")

