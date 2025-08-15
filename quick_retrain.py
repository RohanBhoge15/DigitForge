#!/usr/bin/env python3
"""
Quick retrain with more data but compatible architecture
"""

from train_model import train_neural_network

def main():
    """Quick retrain with better parameters but compatible architecture"""
    print("Quick retraining with more data and better parameters...")
    print("Using compatible 784->64->10 architecture")
    
    # Train with enhanced parameters but compatible architecture
    train_neural_network(
        epochs=200,           # More epochs for better learning
        batch_size=32,        # Smaller batch size
        learning_rate=0.002,  # Lower learning rate
        momentum=0.95,        # Higher momentum
        use_momentum=True
    )
    
    print("Quick retrain complete!")

if __name__ == "__main__":
    main()
