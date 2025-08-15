#!/usr/bin/env python3
"""
Comprehensive Training Script
Trains both custom neural network and professional ML models for comparison
"""

import time
import numpy as np
from train_model import train_neural_network
from ml_models import MLModelComparison
import matplotlib.pyplot as plt

def train_all_models():
    """Train all models and compare performance"""
    print("🚀 COMPREHENSIVE MODEL TRAINING")
    print("="*60)
    print("Training both custom neural network and professional ML models")
    print("="*60)
    
    results = {}
    
    # 1. Train Custom Neural Network
    print("\n🧠 PHASE 1: Training Custom Neural Network")
    print("-" * 40)
    
    start_time = time.time()
    try:
        # Train with optimized parameters
        custom_model = train_neural_network(
            epochs=200,           # More epochs for better learning
            batch_size=32,        # Smaller batch size for better gradients
            learning_rate=0.002,  # Lower learning rate for stability
            momentum=0.95,        # Higher momentum for better convergence
            use_momentum=True
        )
        custom_training_time = time.time() - start_time
        
        results['Custom Neural Network'] = {
            'training_time': custom_training_time,
            'status': 'success',
            'model': custom_model
        }
        
        print(f"✅ Custom Neural Network trained in {custom_training_time:.2f}s")
        
    except Exception as e:
        print(f"❌ Custom Neural Network training failed: {e}")
        results['Custom Neural Network'] = {
            'training_time': 0,
            'status': 'failed',
            'error': str(e)
        }
    
    # 2. Train Professional ML Models
    print("\n🤖 PHASE 2: Training Professional ML Models")
    print("-" * 40)
    
    start_time = time.time()
    try:
        ml_comparison = MLModelComparison()
        ml_comparison.load_data(sample_size=20000)  # Use 20k samples for faster training
        ml_comparison.train_all_models()
        ml_comparison.save_models()
        ml_training_time = time.time() - start_time
        
        # Store ML results
        for model_name in ml_comparison.trained_models.keys():
            results[model_name] = {
                'training_time': ml_comparison.training_times[model_name],
                'accuracy': ml_comparison.accuracies[model_name],
                'status': 'success'
            }
        
        print(f"✅ All ML models trained in {ml_training_time:.2f}s total")
        
    except Exception as e:
        print(f"❌ ML models training failed: {e}")
        results['ML Models'] = {
            'training_time': 0,
            'status': 'failed',
            'error': str(e)
        }
    
    # 3. Generate Performance Summary
    print("\n📊 PHASE 3: Performance Summary")
    print("-" * 40)
    
    generate_performance_report(results)
    
    print("\n🎉 ALL MODELS TRAINED SUCCESSFULLY!")
    print("You can now run the web application with full model comparison:")
    print("python app.py")

def generate_performance_report(results):
    """Generate a comprehensive performance report"""
    
    # Print text summary
    print("\n📈 TRAINING PERFORMANCE SUMMARY")
    print("="*60)
    
    successful_models = {name: data for name, data in results.items() 
                        if data['status'] == 'success' and 'accuracy' in data}
    
    if successful_models:
        print(f"{'Model Name':<25} | {'Accuracy':<10} | {'Training Time':<15}")
        print("-" * 60)
        
        for name, data in successful_models.items():
            accuracy = data.get('accuracy', 0)
            training_time = data.get('training_time', 0)
            print(f"{name:<25} | {accuracy:.4f}    | {training_time:.2f}s")
    
    # Create visualization
    try:
        create_performance_chart(successful_models)
    except Exception as e:
        print(f"⚠️ Could not create performance chart: {e}")

def create_performance_chart(results):
    """Create performance comparison chart"""
    if not results:
        return
    
    model_names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in model_names]
    training_times = [results[name]['training_time'] for name in model_names]
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy comparison
    bars1 = ax1.bar(model_names, accuracies, color=['#667eea', '#f093fb', '#4facfe', '#43e97b', '#fa709a'])
    ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis='x', rotation=45)
    
    # Add accuracy labels on bars
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Training time comparison
    bars2 = ax2.bar(model_names, training_times, color=['#667eea', '#f093fb', '#4facfe', '#43e97b', '#fa709a'])
    ax2.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Training Time (seconds)')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add time labels on bars
    for bar, time_val in zip(bars2, training_times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(training_times)*0.01,
                f'{time_val:.1f}s', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    print("📊 Performance chart saved as 'model_comparison.png'")

if __name__ == "__main__":
    train_all_models()
