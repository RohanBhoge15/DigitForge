# 🧠🤖 DigitForge - Ultimate Digit Recognition Showdown

> **The most comprehensive machine learning comparison project for digit recognition**
>
> Build neural networks from scratch, compare with deep learning CNNs, and battle industry-standard algorithms in real-time through a stunning web interface.

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg?style=flat-square)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange.svg?style=flat-square)
![NumPy](https://img.shields.io/badge/NumPy-From%20Scratch-yellow.svg?style=flat-square)
![Flask](https://img.shields.io/badge/Flask-Web%20App-green.svg?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg?style=flat-square)

[Features](#-features) • [Quick Start](#-quick-start) • [Models](#-the-three-champions) • [Architecture](#-project-structure) • [Demo](#-web-interface)

</div>

---

## 🌟 Features

### 🎯 Three Approaches to Digit Recognition

| Approach | Technology | Accuracy | Speed | Best For |
|----------|-----------|----------|-------|----------|
| **🧠 Custom Neural Network** | Pure NumPy | ~96-97% | ⚡ Fast | Learning ML fundamentals |
| **🔥 Deep Learning CNN** | TensorFlow/Keras | **~99%+** 🏆 | 🐢 Slower | Production accuracy |
| **🤖 Professional ML** | scikit-learn | ~97-98% | ⚡⚡ Fastest | Fast inference |

### 🎨 Interactive Features

- ✏️ **Draw Digits** - Natural HTML5 canvas with smooth pen strokes (mobile-friendly)
- 🎯 **Instant Predictions** - See predictions from any model in real-time
- ⚡ **Model Comparison** - Compare all 8+ algorithms side-by-side on your drawing
- 📊 **Performance Dashboard** - Interactive charts showing accuracy, speed, and model sizes
- 🔍 **Neural Network Visualization** - Watch neurons activate as your digit is processed
- 📈 **Detailed Metrics** - Comprehensive benchmarking across all models

### 🎓 Educational Focus

- **Mathematical Clarity** - See exactly how backpropagation works
- **Multiple Implementations** - Compare from-scratch vs framework approaches
- **Trade-offs Analysis** - Accuracy vs Speed vs Model Size comparisons
- **Well-Documented Code** - Clean, commented implementations with explanations

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip (or conda)

### Installation

```bash
# Clone the repository
git clone https://github.com/RohanBhoge15/DigitForge.git
cd DigitForge

# Install dependencies
pip install -r requirements.txt
```

### Option 1: Complete Setup (Recommended)

```bash
# Train all models and generate metrics
python train_all_models.py

# Start the web application
python app.py
```

Then open your browser to `http://localhost:5000`

### Option 2: Custom Setup

```bash
# Train only what you need
python quick_retrain.py       # Custom Neural Network
python cnn_model.py           # Deep Learning CNN
python ml_models.py           # Professional ML algorithms
python metrics_comparison.py  # Generate benchmarks

# Start the app
python app.py
```

---

## 🏗️ Project Structure

```
DigitForge/
│
├── 🧠 Model Implementations
│   ├── neural_network.py          # Pure NumPy neural network
│   ├── cnn_model.py               # TensorFlow/Keras CNN
│   └── ml_models.py               # scikit-learn algorithms (RF, SVM, GB, KNN, LR)
│
├── 🏭 Training & Benchmarking
│   ├── train_model.py             # Basic training script
│   ├── quick_retrain.py           # Fast neural network retraining
│   ├── train_all_models.py        # Master training script
│   └── metrics_comparison.py      # Comprehensive benchmarking
│
├── 🌐 Web Application
│   ├── app.py                     # Flask server
│   ├── templates/
│   │   ├── index.html             # Main drawing interface
│   │   └── metrics.html           # Performance dashboard
│   └── static/
│       ├── script.js              # Interactive JavaScript
│       └── style.css              # Modern styling
│
├── 📦 Generated Assets (in .gitignore)
│   ├── trained_model.pkl          # Custom NN weights
│   ├── cnn_model.h5               # CNN model
│   ├── ml_models.pkl              # Professional ML models
│   ├── comparison_metrics.json    # Benchmark results
│   └── *.png                      # Generated plots
│
└── 📋 Configuration
    ├── requirements.txt           # Python dependencies
    ├── .gitignore                 # Git ignore patterns
    └── README.md                  # This file
```

---

## 🧠 The Three Champions

### 1️⃣ Custom Neural Network (Pure NumPy)

**The Educational Champion** - See machine learning from first principles

```
INPUT LAYER      →  784 neurons (28×28 pixels)
HIDDEN LAYER     →   64 neurons (ReLU activation)
OUTPUT LAYER     →   10 neurons (Softmax)
```

**Training**: SGD with momentum optimization
**Key Advantage**: Understand every mathematical operation
**Learning Resource**: Perfect for understanding backpropagation

---

### 2️⃣ Convolutional Neural Network (TensorFlow/Keras)

**The Accuracy Champion** - Deep learning for maximum performance

```
CONV BLOCK 1  →  32 filters (3×3) + ReLU + MaxPool + BatchNorm
CONV BLOCK 2  →  64 filters (3×3) + ReLU + MaxPool + BatchNorm
CONV BLOCK 3  →  64 filters (3×3) + ReLU + BatchNorm
DENSE LAYER   →  128 neurons + Dropout(0.5)
OUTPUT        →  10 neurons (Softmax)
```

**Training**: Adam optimizer with categorical crossentropy
**Key Advantage**: State-of-the-art accuracy (~99%+)
**Production Ready**: Can handle real-world digit recognition

---

### 3️⃣ Professional ML Arsenal (scikit-learn)

**The Pragmatist's Choice** - Battle-tested algorithms

- **Random Forest** - Ensemble of decision trees
- **SVM (RBF)** - Powerful non-linear classifier
- **Gradient Boosting** - Sequential improvement method
- **K-Nearest Neighbors** - Instance-based learning
- **Logistic Regression** - Baseline linear classifier

**Key Advantage**: Fast inference, interpretable, minimal dependencies
**Use Case**: Production systems where speed matters more than micro-optimizations

---

## 📊 Performance Benchmarks

Run `python metrics_comparison.py` to generate comprehensive benchmarks:

### Typical Results

| Metric | Custom NN | CNN | Random Forest | SVM | Gradient Boosting |
|--------|-----------|-----|---------------|-----|------------------|
| Accuracy | 96-97% | **99%+** ⭐ | 96-97% | 97-98% | 97-98% |
| Training Time | 2-5m | 5-15m | 2-3m | 3-5m | 2-3m |
| Inference Speed | ~1ms | ~2-3ms | <0.1ms | <0.1ms | <0.1ms |
| Model Size | 200KB | 5MB | 150MB | 50-100MB | 100MB+ |
| Ease of Learning | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ |

---

## 🌐 Web Interface

### Main Drawing Arena (`/`)

1. **Draw a digit** using your mouse (or touch on mobile)
2. **Select a model** from the dropdown
3. **Get predictions** with confidence scores
4. **Visualize** the neural network's decision process
5. **Compare** all models at once

### Features:

- 🎨 **Canvas Area** - Large, responsive drawing space
- 🔄 **Model Switcher** - Instantly switch between 8+ models
- 📈 **Confidence Chart** - See probabilities for all digits (0-9)
- 🧠 **Neuron Activation Map** - Watch the neural network process your drawing
- 🏆 **Model Comparison** - Side-by-side predictions from all models

### Performance Dashboard (`/metrics_dashboard`)

- 📊 **Accuracy Comparison** - Bar charts comparing all models
- ⚡ **Speed Analysis** - Inference time comparisons
- 💾 **Model Size** - Memory footprint of each model
- 🎯 **Training Time** - Time required to train each model
- 📋 **Detailed Metrics Table** - Raw benchmark data

---

## 🛠️ Technical Highlights

### Image Preprocessing

The app intelligently preprocesses your drawings:

1. **Color Inversion** - Matches MNIST's black-on-white format
2. **Bounding Box Detection** - Finds the actual digit
3. **Smart Resizing** - Maintains aspect ratio
4. **Centering** - Places digit in center of 28×28 image
5. **Smoothing** - Gaussian blur for better recognition
6. **Normalization** - Scales pixel values to [0, 1]

### Prediction Pipeline

- **Custom NN**: Forward pass through 2 layers of pure NumPy math
- **CNN**: TensorFlow inference on GPU (if available)
- **ML Models**: scikit-learn prediction
- **Batch Processing**: Handle multiple predictions efficiently

### Error Handling

- Validates that drawings contain actual digits (not random scribbles)
- Provides helpful feedback when input is unclear
- Graceful fallbacks if models aren't loaded

---

## 📦 Dependencies

```
numpy>=1.21.0              # Numerical computing
matplotlib>=3.5.0          # Data visualization
scikit-learn>=1.0.0        # Professional ML algorithms
pandas>=1.3.0              # Data manipulation
flask>=2.0.0               # Web framework
opencv-python>=4.5.0       # Image processing
pillow>=8.3.0              # PIL Image library
seaborn>=0.11.0            # Statistical visualization
tensorflow>=2.10.0         # Deep learning framework
plotly>=5.0.0              # Interactive plots
```

---

## 🎓 Learning Resources

### Understanding the Custom Neural Network

The custom implementation is heavily documented. Key files:

- **`neural_network.py`** - Forward pass, backward pass, activation functions
- **`train_model.py`** - Training loop with momentum-based SGD
- Look for comments like `# 🧠 Mathematical Note:` for explanations

### Understanding CNNs

- See how convolution layers extract features
- Understand pooling and batch normalization
- Compare architecture choices with performance

### Comparing Approaches

Run all three and observe:
- Where each approach succeeds
- Trade-offs between complexity and accuracy
- Why certain algorithms are production-standard

---

## 🚀 Advanced Usage

### Custom Model Training

```python
from neural_network import NeuralNetwork

# Create and train
nn = NeuralNetwork(input_size=784, hidden_size=128, output_size=10)
nn.train(X_train, y_train, epochs=50, learning_rate=0.01)

# Make predictions
predictions = nn.predict(X_test)
```

### Using in Your Own Project

```python
import pickle
from neural_network import NeuralNetwork

# Load the trained model
with open('trained_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Make predictions
prediction = model.predict(image_data)
```

---

## 🤝 Contributing

Found an issue? Want to improve the project?

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **MNIST Dataset** - Handwritten digit dataset used for training
- **NumPy, TensorFlow, scikit-learn** - Amazing open-source libraries
- **The ML Community** - For inspiration and best practices

---

## 📧 Contact

Have questions or suggestions? Feel free to reach out!

- GitHub: [@RohanBhoge15](https://github.com/RohanBhoge15)
- Email: Open an issue in the repository

---

<div align="center">

### ⭐ If you find this project helpful, please consider giving it a star! ⭐

**Built with ❤️ for the machine learning community**

Made to educate, inspire, and compete 🚀

</div>
