"""
Comprehensive Metrics Comparison System
Compares CNN, Custom Neural Network, and ML Models across multiple dimensions
"""

import numpy as np
import pickle
import time
import os
import json
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from neural_network import NeuralNetwork
from ml_models import MLModelComparison
from cnn_model import CNNModel


class MetricsComparison:
    def __init__(self):
        """Initialize metrics comparison system"""
        self.models = {}
        self.metrics = {
            'training_time': {},
            'test_accuracy': {},
            'prediction_speed': {},
            'model_size': {},
            'parameters': {}
        }
        self.test_data = None
        self.test_labels = None
        
    def load_test_data(self, sample_size=1000):
        """Load test data for evaluation"""
        print("Loading test data for metrics comparison...")
        
        # Load MNIST
        mnist = fetch_openml('mnist_784', version=1, parser='auto')
        X, y = mnist.data.to_numpy(), mnist.target.to_numpy().astype(int)
        
        # Split and get test set
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Use subset for faster evaluation
        if sample_size and sample_size < len(X_test):
            indices = np.random.choice(len(X_test), sample_size, replace=False)
            X_test = X_test[indices]
            y_test = y_test[indices]
        
        # Normalize
        self.test_data = X_test / 255.0
        self.test_labels = y_test
        
        print(f"✅ Loaded {len(self.test_data)} test samples")
        
    def load_all_models(self):
        """Load all trained models"""
        print("\n📦 Loading all models...")
        
        # Load Custom Neural Network
        try:
            with open('trained_model.pkl', 'rb') as f:
                model_data = pickle.load(f)
            if isinstance(model_data, NeuralNetwork):
                self.models['Custom Neural Network'] = model_data
                print("✅ Custom Neural Network loaded")
            else:
                nn = NeuralNetwork()
                nn.load_model('trained_model.pkl')
                self.models['Custom Neural Network'] = nn
                print("✅ Custom Neural Network loaded (legacy)")
        except Exception as e:
            print(f"❌ Failed to load Custom Neural Network: {e}")
        
        # Load CNN
        try:
            cnn = CNNModel()
            if cnn.load_model():
                self.models['CNN'] = cnn
                print("✅ CNN loaded")
        except Exception as e:
            print(f"❌ Failed to load CNN: {e}")
        
        # Load ML Models
        try:
            ml_models = MLModelComparison()
            if ml_models.load_models('ml_models.pkl'):
                for model_name in ml_models.trained_models.keys():
                    self.models[model_name] = ml_models.trained_models[model_name]
                self.ml_comparison = ml_models
                print(f"✅ Loaded {len(ml_models.trained_models)} ML models")
        except Exception as e:
            print(f"❌ Failed to load ML models: {e}")
        
        print(f"\n📊 Total models loaded: {len(self.models)}")
        
    def measure_prediction_speed(self, num_samples=100):
        """Measure prediction speed for each model"""
        print("\n⚡ Measuring prediction speed...")
        
        if self.test_data is None:
            self.load_test_data(sample_size=num_samples)
        
        # Get sample data
        sample_data = self.test_data[:num_samples]
        
        for model_name, model in self.models.items():
            try:
                start_time = time.time()
                
                if model_name == 'CNN':
                    # CNN needs reshaped input
                    for i in range(num_samples):
                        _ = model.predict(sample_data[i:i+1])
                elif model_name == 'Custom Neural Network':
                    # Custom NN
                    for i in range(num_samples):
                        _ = model.predict(sample_data[i:i+1])
                else:
                    # ML models
                    for i in range(num_samples):
                        _ = model.predict(sample_data[i:i+1])
                
                elapsed_time = time.time() - start_time
                avg_time_ms = (elapsed_time / num_samples) * 1000
                
                self.metrics['prediction_speed'][model_name] = avg_time_ms
                print(f"  {model_name}: {avg_time_ms:.2f} ms/sample")
                
            except Exception as e:
                print(f"  ❌ {model_name}: Error - {e}")
                self.metrics['prediction_speed'][model_name] = None
    
    def measure_accuracy(self, num_samples=1000):
        """Measure test accuracy for each model"""
        print("\n🎯 Measuring test accuracy...")
        
        if self.test_data is None:
            self.load_test_data(sample_size=num_samples)
        
        sample_data = self.test_data[:num_samples]
        sample_labels = self.test_labels[:num_samples]
        
        for model_name, model in self.models.items():
            try:
                predictions = []
                
                if model_name == 'CNN':
                    for i in range(num_samples):
                        pred = model.predict(sample_data[i:i+1])
                        predictions.append(pred)
                elif model_name == 'Custom Neural Network':
                    for i in range(num_samples):
                        pred = model.predict(sample_data[i:i+1])[0]
                        predictions.append(pred)
                else:
                    for i in range(num_samples):
                        pred = model.predict(sample_data[i:i+1])[0]
                        predictions.append(pred)
                
                predictions = np.array(predictions)
                accuracy = np.mean(predictions == sample_labels)
                
                self.metrics['test_accuracy'][model_name] = accuracy
                print(f"  {model_name}: {accuracy*100:.2f}%")
                
            except Exception as e:
                print(f"  ❌ {model_name}: Error - {e}")
                self.metrics['test_accuracy'][model_name] = None
    
    def get_model_sizes(self):
        """Get model file sizes"""
        print("\n💾 Measuring model sizes...")
        
        model_files = {
            'Custom Neural Network': 'trained_model.pkl',
            'CNN': 'cnn_model.h5',
            'ML Models': 'ml_models.pkl'
        }
        
        for model_name, file_path in model_files.items():
            if os.path.exists(file_path):
                size_bytes = os.path.getsize(file_path)
                size_mb = size_bytes / (1024 * 1024)
                self.metrics['model_size'][model_name] = size_mb
                print(f"  {model_name}: {size_mb:.2f} MB")
            else:
                self.metrics['model_size'][model_name] = None
    
    def load_training_times(self):
        """Load training times from saved metrics"""
        print("\n⏱️ Loading training times...")
        
        # Custom NN - from training history
        try:
            with open('trained_model.pkl', 'rb') as f:
                model_data = pickle.load(f)
            # Training time not directly stored, estimate from history
            self.metrics['training_time']['Custom Neural Network'] = None
            print("  Custom Neural Network: Not recorded")
        except:
            pass
        
        # CNN
        try:
            with open('cnn_metrics.pkl', 'rb') as f:
                cnn_metrics = pickle.load(f)
            self.metrics['training_time']['CNN'] = cnn_metrics.get('training_time', None)
            print(f"  CNN: {cnn_metrics.get('training_time', 0):.2f}s")
        except:
            print("  CNN: Not recorded")
        
        # ML Models
        try:
            ml_models = MLModelComparison()
            if ml_models.load_models('ml_models.pkl'):
                for model_name, train_time in ml_models.training_times.items():
                    self.metrics['training_time'][model_name] = train_time
                    print(f"  {model_name}: {train_time:.2f}s")
        except:
            print("  ML Models: Not recorded")
    
    def generate_comparison_report(self):
        """Generate comprehensive comparison report"""
        print("\n" + "="*60)
        print("📊 COMPREHENSIVE MODEL COMPARISON REPORT")
        print("="*60)
        
        # Load all metrics
        self.load_all_models()
        self.load_test_data(sample_size=1000)
        self.measure_accuracy(num_samples=1000)
        self.measure_prediction_speed(num_samples=100)
        self.get_model_sizes()
        self.load_training_times()
        
        # Save metrics
        self.save_metrics()
        
        # Generate visualizations
        self.create_comparison_plots()
        
        # Print summary
        self.print_summary()
        
    def save_metrics(self):
        """Save all metrics to file"""
        metrics_file = 'comparison_metrics.pkl'
        with open(metrics_file, 'wb') as f:
            pickle.dump(self.metrics, f)
        print(f"\n✅ Metrics saved to {metrics_file}")
        
        # Also save as JSON for web display
        json_metrics = {}
        for metric_type, values in self.metrics.items():
            json_metrics[metric_type] = {
                k: float(v) if v is not None else None 
                for k, v in values.items()
            }
        
        with open('comparison_metrics.json', 'w') as f:
            json.dump(json_metrics, f, indent=2)
        print(f"✅ Metrics saved to comparison_metrics.json")

    def create_comparison_plots(self):
        """Create comprehensive comparison visualizations"""
        print("\n📈 Generating comparison plots...")

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (16, 10)

        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Comprehensive Model Comparison: CNN vs Custom NN vs ML Models',
                     fontsize=16, fontweight='bold')

        # 1. Accuracy Comparison
        ax1 = axes[0, 0]
        accuracy_data = {k: v*100 for k, v in self.metrics['test_accuracy'].items() if v is not None}
        if accuracy_data:
            colors = ['#FF6B6B' if 'CNN' in k else '#4ECDC4' if 'Custom' in k else '#95E1D3'
                     for k in accuracy_data.keys()]
            bars = ax1.bar(range(len(accuracy_data)), list(accuracy_data.values()), color=colors)
            ax1.set_xticks(range(len(accuracy_data)))
            ax1.set_xticklabels(list(accuracy_data.keys()), rotation=45, ha='right')
            ax1.set_ylabel('Accuracy (%)')
            ax1.set_title('Test Accuracy Comparison', fontweight='bold')
            ax1.set_ylim([0, 100])

            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, accuracy_data.values())):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')

        # 2. Prediction Speed Comparison
        ax2 = axes[0, 1]
        speed_data = {k: v for k, v in self.metrics['prediction_speed'].items() if v is not None}
        if speed_data:
            colors = ['#FF6B6B' if 'CNN' in k else '#4ECDC4' if 'Custom' in k else '#95E1D3'
                     for k in speed_data.keys()]
            bars = ax2.bar(range(len(speed_data)), list(speed_data.values()), color=colors)
            ax2.set_xticks(range(len(speed_data)))
            ax2.set_xticklabels(list(speed_data.keys()), rotation=45, ha='right')
            ax2.set_ylabel('Time (ms)')
            ax2.set_title('Prediction Speed (Lower is Better)', fontweight='bold')

            # Add value labels
            for bar, val in zip(bars, speed_data.values()):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(speed_data.values())*0.02,
                        f'{val:.2f}ms', ha='center', va='bottom', fontweight='bold', fontsize=9)

        # 3. Training Time Comparison
        ax3 = axes[0, 2]
        train_time_data = {k: v for k, v in self.metrics['training_time'].items() if v is not None}
        if train_time_data:
            colors = ['#FF6B6B' if 'CNN' in k else '#4ECDC4' if 'Custom' in k else '#95E1D3'
                     for k in train_time_data.keys()]
            bars = ax3.bar(range(len(train_time_data)), list(train_time_data.values()), color=colors)
            ax3.set_xticks(range(len(train_time_data)))
            ax3.set_xticklabels(list(train_time_data.keys()), rotation=45, ha='right')
            ax3.set_ylabel('Time (seconds)')
            ax3.set_title('Training Time (Lower is Better)', fontweight='bold')

            # Add value labels
            for bar, val in zip(bars, train_time_data.values()):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(train_time_data.values())*0.02,
                        f'{val:.1f}s', ha='center', va='bottom', fontweight='bold', fontsize=9)

        # 4. Model Size Comparison
        ax4 = axes[1, 0]
        size_data = {k: v for k, v in self.metrics['model_size'].items() if v is not None}
        if size_data:
            colors = ['#FF6B6B' if 'CNN' in k else '#4ECDC4' if 'Custom' in k else '#95E1D3'
                     for k in size_data.keys()]
            bars = ax4.bar(range(len(size_data)), list(size_data.values()), color=colors)
            ax4.set_xticks(range(len(size_data)))
            ax4.set_xticklabels(list(size_data.keys()), rotation=45, ha='right')
            ax4.set_ylabel('Size (MB)')
            ax4.set_title('Model Size (Lower is Better)', fontweight='bold')

            # Add value labels
            for bar, val in zip(bars, size_data.values()):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(size_data.values())*0.02,
                        f'{val:.2f}MB', ha='center', va='bottom', fontweight='bold', fontsize=9)

        # 5. Accuracy vs Speed Scatter Plot
        ax5 = axes[1, 1]
        if accuracy_data and speed_data:
            # Get common models
            common_models = set(accuracy_data.keys()) & set(speed_data.keys())
            x_vals = [speed_data[m] for m in common_models]
            y_vals = [accuracy_data[m] for m in common_models]
            colors_scatter = ['#FF6B6B' if 'CNN' in m else '#4ECDC4' if 'Custom' in m else '#95E1D3'
                            for m in common_models]

            ax5.scatter(x_vals, y_vals, c=colors_scatter, s=200, alpha=0.6, edgecolors='black', linewidth=2)

            # Add labels
            for i, model in enumerate(common_models):
                ax5.annotate(model, (x_vals[i], y_vals[i]),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)

            ax5.set_xlabel('Prediction Speed (ms) - Lower is Better')
            ax5.set_ylabel('Accuracy (%) - Higher is Better')
            ax5.set_title('Accuracy vs Speed Trade-off', fontweight='bold')
            ax5.grid(True, alpha=0.3)

        # 6. Performance Summary Table
        ax6 = axes[1, 2]
        ax6.axis('off')

        # Create summary text
        summary_text = "🏆 PERFORMANCE SUMMARY\n\n"

        # Best accuracy
        if accuracy_data:
            best_acc_model = max(accuracy_data.items(), key=lambda x: x[1])
            summary_text += f"🎯 Best Accuracy:\n{best_acc_model[0]}\n{best_acc_model[1]:.2f}%\n\n"

        # Fastest prediction
        if speed_data:
            fastest_model = min(speed_data.items(), key=lambda x: x[1])
            summary_text += f"⚡ Fastest Prediction:\n{fastest_model[0]}\n{fastest_model[1]:.2f}ms\n\n"

        # Fastest training
        if train_time_data:
            fastest_train = min(train_time_data.items(), key=lambda x: x[1])
            summary_text += f"🏋️ Fastest Training:\n{fastest_train[0]}\n{fastest_train[1]:.1f}s\n\n"

        # Smallest model
        if size_data:
            smallest_model = min(size_data.items(), key=lambda x: x[1])
            summary_text += f"💾 Smallest Model:\n{smallest_model[0]}\n{smallest_model[1]:.2f}MB"

        ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig('comprehensive_model_comparison.png', dpi=300, bbox_inches='tight')
        print("✅ Comparison plots saved to 'comprehensive_model_comparison.png'")

    def print_summary(self):
        """Print detailed summary to console"""
        print("\n" + "="*60)
        print("📋 DETAILED COMPARISON SUMMARY")
        print("="*60)

        # Accuracy
        print("\n🎯 TEST ACCURACY:")
        for model, acc in sorted(self.metrics['test_accuracy'].items(),
                                key=lambda x: x[1] if x[1] else 0, reverse=True):
            if acc is not None:
                print(f"  {model:<30} {acc*100:>6.2f}%")

        # Prediction Speed
        print("\n⚡ PREDICTION SPEED (ms per sample):")
        for model, speed in sorted(self.metrics['prediction_speed'].items(),
                                   key=lambda x: x[1] if x[1] else float('inf')):
            if speed is not None:
                print(f"  {model:<30} {speed:>6.2f} ms")

        # Training Time
        print("\n⏱️ TRAINING TIME:")
        for model, time_val in sorted(self.metrics['training_time'].items(),
                                      key=lambda x: x[1] if x[1] else float('inf')):
            if time_val is not None:
                print(f"  {model:<30} {time_val:>6.1f} s")

        # Model Size
        print("\n💾 MODEL SIZE:")
        for model, size in sorted(self.metrics['model_size'].items(),
                                 key=lambda x: x[1] if x[1] else float('inf')):
            if size is not None:
                print(f"  {model:<30} {size:>6.2f} MB")

        print("\n" + "="*60)


if __name__ == "__main__":
    print("🚀 Comprehensive Model Metrics Comparison")
    print("="*60)

    comparison = MetricsComparison()
    comparison.generate_comparison_report()

    print("\n🎉 Comparison complete!")
    print("📊 Check 'comprehensive_model_comparison.png' for visualizations")
    print("📁 Metrics saved to 'comparison_metrics.json'")

