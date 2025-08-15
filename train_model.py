import numpy as np
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork, load_mnist_data
import time
import os

def train_neural_network(epochs=100, batch_size=128, learning_rate=0.01, momentum=0.9, use_momentum=True):
    """
    Train the neural network on MNIST dataset
    """
    print("=" * 60)
    print("NEURAL NETWORK TRAINING FROM SCRATCH")
    print("=" * 60)
    
    # Load MNIST data
    X_train, X_test, y_train, y_test, y_train_labels, y_test_labels = load_mnist_data()
    
    # Initialize neural network
    print(f"\nInitializing Neural Network:")
    print(f"Architecture: {X_train.shape[1]} -> 64 -> 10")
    print(f"Learning Rate: {learning_rate}")
    print(f"Momentum: {momentum}")
    print(f"Batch Size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Using Momentum: {use_momentum}")
    
    nn = NeuralNetwork(
        input_size=X_train.shape[1],
        hidden_size=64,
        output_size=10,
        learning_rate=learning_rate,
        momentum=momentum
    )
    
    # Training loop
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    
    start_time = time.time()
    n_batches = len(X_train) // batch_size
    
    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss = 0
        
        # Shuffle training data
        indices = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        
        # Mini-batch training
        for batch in range(n_batches):
            start_idx = batch * batch_size
            end_idx = start_idx + batch_size
            
            X_batch = X_train_shuffled[start_idx:end_idx]
            y_batch = y_train_shuffled[start_idx:end_idx]
            
            # Forward pass
            y_pred = nn.forward_pass(X_batch)
            
            # Compute loss
            batch_loss = nn.compute_loss(y_batch, y_pred)
            epoch_loss += batch_loss
            
            # Backward pass
            dW1, db1, dW2, db2 = nn.backward_pass(X_batch, y_batch, y_pred)
            
            # Update weights
            nn.optimize(dW1, db1, dW2, db2, use_momentum=use_momentum)
        
        # Calculate average loss for epoch
        avg_loss = epoch_loss / n_batches
        nn.loss_history.append(avg_loss)
        
        # Evaluate on training and test sets every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            train_accuracy = nn.evaluate(X_train[:1000], y_train[:1000])  # Sample for speed
            test_accuracy = nn.evaluate(X_test[:1000], y_test[:1000])    # Sample for speed
            nn.accuracy_history.append(test_accuracy)
            
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Train Acc: {train_accuracy:.4f} | "
                  f"Test Acc: {test_accuracy:.4f} | "
                  f"Time: {epoch_time:.2f}s")
        else:
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s")
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.2f} seconds")
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    
    train_accuracy = nn.evaluate(X_train, y_train)
    test_accuracy = nn.evaluate(X_test, y_test)
    
    print(f"Final Training Accuracy: {train_accuracy:.4f}")
    print(f"Final Test Accuracy: {test_accuracy:.4f}")
    
    # Save the trained model
    model_path = "trained_model.pkl"
    nn.save_model(model_path)
    
    # Plot training history
    plot_training_history(nn)
    
    return nn

def plot_training_history(nn):
    """
    Plot training loss and accuracy history
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    ax1.plot(nn.loss_history)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cross-Entropy Loss')
    ax1.grid(True)
    
    # Plot accuracy (only available every 10 epochs)
    if nn.accuracy_history:
        epochs_recorded = range(10, len(nn.accuracy_history) * 10 + 1, 10)
        ax2.plot(epochs_recorded, nn.accuracy_history)
        ax2.set_title('Test Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Training history plot saved as 'training_history.png'")

def test_predictions(nn, X_test, y_test_labels, num_samples=10):
    """
    Test predictions on random samples and display results
    """
    print("\n" + "=" * 60)
    print("SAMPLE PREDICTIONS")
    print("=" * 60)
    
    # Get random samples
    indices = np.random.choice(len(X_test), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        sample = X_test[idx:idx+1]
        true_label = y_test_labels[idx]
        
        # Get prediction and probabilities
        prediction = nn.predict(sample)[0]
        probabilities = nn.predict_proba(sample)[0]
        confidence = np.max(probabilities)
        
        print(f"Sample {i+1:2d}: True={true_label}, Predicted={prediction}, "
              f"Confidence={confidence:.3f} {'✓' if prediction == true_label else '✗'}")

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Train the model
    trained_model = train_neural_network(
        epochs=50,
        batch_size=128,
        learning_rate=0.01,
        momentum=0.9,
        use_momentum=True
    )
    
    # Load test data for final testing
    _, X_test, _, _, _, y_test_labels = load_mnist_data()
    
    # Test some predictions
    test_predictions(trained_model, X_test, y_test_labels, num_samples=10)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("Model saved as 'trained_model.pkl'")
    print("You can now run the Flask app to test with custom images!")
    print("=" * 60)
