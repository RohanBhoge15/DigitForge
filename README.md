# Neural Network from Scratch - MNIST Digit Recognition

A complete implementation of a neural network built from scratch using only NumPy, with a Flask web interface for testing digit recognition.

## 🧠 Project Overview

This project implements a 3-layer neural network (784 → 64 → 10) from scratch to recognize handwritten digits from the MNIST dataset. The network uses:

- **Input Layer**: 784 nodes (28×28 flattened pixel values)
- **Hidden Layer**: 64 nodes with ReLU activation
- **Output Layer**: 10 nodes with Softmax activation (for digit probabilities 0-9)

### Key Features

- ✅ **Pure NumPy Implementation**: No deep learning frameworks used
- ✅ **Forward Propagation**: Sequential dot operations with activation functions
- ✅ **Backpropagation**: Mathematical derivatives for gradient computation
- ✅ **SGD with Momentum**: Optimized training with momentum updates
- ✅ **Web Interface**: Flask app with drawing canvas and image upload
- ✅ **Smart Error Handling**: Detects non-digit images
- ✅ **Real-time Visualization**: Probability distributions for all digits

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python train_model.py
```

This will:
- Download and preprocess the MNIST dataset
- Train the neural network for 50 epochs
- Save the trained model as `trained_model.pkl`
- Display training progress and final accuracy
- Generate a training history plot

### 3. Run the Web Application

```bash
python app.py
```

Open your browser and go to: `http://localhost:5000`

## 📁 Project Structure

```
├── neural_network.py      # Core neural network implementation
├── train_model.py         # Training script with SGD and momentum
├── app.py                # Flask web application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── templates/
│   └── index.html        # Web interface HTML
├── static/
│   ├── style.css         # Styling and animations
│   └── script.js         # Frontend JavaScript logic
├── trained_model.pkl     # Saved model (generated after training)
└── training_history.png  # Training visualization (generated)
```

## 🔬 Neural Network Architecture

### Network Design
- **Input**: 784 features (28×28 pixel values normalized to [0,1])
- **Hidden**: 64 neurons with ReLU activation
- **Output**: 10 neurons with Softmax activation (digit probabilities)

### Mathematical Operations
1. **Forward Pass**: 
   - `z1 = X·W1 + b1`
   - `a1 = ReLU(z1)`
   - `z2 = a1·W2 + b2`
   - `a2 = Softmax(z2)`

2. **Loss Function**: Cross-entropy loss
3. **Backpropagation**: Chain rule for gradient computation
4. **Optimization**: SGD with momentum

### Training Process
- **Batch Size**: 128 samples
- **Learning Rate**: 0.01
- **Momentum**: 0.9
- **Epochs**: 50 (configurable)
- **Weight Initialization**: Xavier initialization

## 🌐 Web Interface Features

### Drawing Canvas
- Draw digits directly on a 280×280 canvas
- Automatic preprocessing to 28×28 grayscale
- Clear canvas functionality
- Touch support for mobile devices

### Image Upload
- Upload image files (PNG, JPG, etc.)
- Automatic resizing and preprocessing
- Preview uploaded images
- Format conversion to match MNIST data

### Smart Predictions
- Real-time digit recognition
- Confidence scores for predictions
- Probability distribution for all 10 digits
- Error detection for non-digit images
- Visual feedback with animations

### 🧠 Neural Network Visualization
- **Input Layer Visualization**: Shows up to 50 most activated neurons (pixels)
- **Hidden Layer Visualization**: Displays up to 25 most activated hidden neurons
- **Output Layer Visualization**: All 10 digit probabilities with confidence scores
- **Real-time Activation Mapping**: See exactly which neurons fire for your drawing
- **Activation Thresholds**: Sensitive detection (>0.001) to show more neuron activity

### Error Handling
- Detects images that don't contain digits
- Handles invalid image formats
- Network error recovery
- User-friendly error messages

## 📊 Performance & Results

### Achieved Performance (Custom Implementation)
- **Training Accuracy**: **96.67%** (after 200 epochs)
- **Test Accuracy**: **~90-95%** (varies by test set)
- **Training Time**: ~5-8 minutes (200 epochs)
- **Model Size**: ~200KB (pure NumPy weights)

### Training Configuration Used
- **Epochs**: 200 (optimized for better convergence)
- **Learning Rate**: 0.002 (reduced for stability)
- **Momentum**: 0.95 (increased for better convergence)
- **Batch Size**: 32 (smaller for better gradients)
- **Optimizer**: SGD with Momentum
- **Loss Function**: Cross-entropy
- **Weight Initialization**: Xavier/Glorot initialization

### Real-World Performance Characteristics
- **Clear, well-drawn digits**: 70-90% confidence
- **Ambiguous drawings**: 20-40% confidence (appropriate uncertainty)
- **Complex/messy digits**: 15-30% confidence
- **Non-digits**: Correctly rejected by validation

### Performance vs Built-in Models
Our custom implementation achieves reasonable results but is intentionally educational:
- **Custom Neural Network**: ~20-40% typical confidence, some misclassifications
- **Production Models** (CNN/TensorFlow): 95-99% accuracy, 90%+ confidence
- **Trade-off**: Learning value vs performance optimization

## 🛠️ Technical Implementation

### Core Components

1. **NeuralNetwork Class** (`neural_network.py`):
   - Forward and backward propagation
   - Activation functions (ReLU, Sigmoid, Softmax)
   - SGD with momentum optimization
   - Model persistence (save/load)

2. **Training Pipeline** (`train_model.py`):
   - MNIST data loading and preprocessing
   - Mini-batch training with shuffling
   - Progress tracking and visualization
   - Model evaluation and saving

3. **Web Application** (`app.py`):
   - Flask REST API for predictions
   - Image preprocessing pipeline
   - Model loading and inference
   - Error handling and validation
   - **Neural network activation extraction**

4. **Frontend** (`templates/`, `static/`):
   - Interactive drawing canvas
   - File upload with preview
   - Real-time prediction display
   - **Neural network visualization**
   - Responsive design with animations

### 🧠 Neural Network Architecture Deep Dive

#### Layer Structure
```
Input Layer (784 neurons)    Hidden Layer (64 neurons)    Output Layer (10 neurons)
     [28x28 pixels]     →         [ReLU activation]    →      [Softmax probabilities]
         ↓                              ↓                            ↓
   Flattened to 784         Feature extraction &           Digit probabilities
   grayscale values         pattern recognition              (0-9 classes)
```

#### Activation Flow
1. **Input Processing**: 28×28 image → 784 flattened pixels (normalized 0-1)
2. **Hidden Layer**: `ReLU(X·W1 + b1)` - extracts features and patterns
3. **Output Layer**: `Softmax(H·W2 + b2)` - converts to digit probabilities

#### Weight Matrices
- **W1**: 784×64 matrix (input to hidden connections)
- **W2**: 64×10 matrix (hidden to output connections)
- **Total Parameters**: ~50,000 trainable weights and biases

### Key Algorithms

- **Xavier Weight Initialization**: Prevents vanishing/exploding gradients
- **Mini-batch SGD**: Efficient training with momentum
- **Cross-entropy Loss**: Optimal for multi-class classification
- **Softmax Activation**: Converts logits to probabilities
- **Image Preprocessing**: Normalization and format standardization

### 🔍 Neuron Activation Extraction Process

#### How Neural Network Visualization Works

1. **Forward Pass Capture**:
   ```python
   # Input layer: Raw pixel values (784 neurons)
   input_layer = processed_image.flatten()

   # Hidden layer: After ReLU activation (64 neurons)
   z1 = np.dot(processed_image, model.W1) + model.b1
   hidden_layer = model.relu(z1).flatten()

   # Output layer: After softmax (10 neurons)
   z2 = np.dot(hidden_layer.reshape(1, -1), model.W2) + model.b2
   output_layer = model.softmax(z2).flatten()
   ```

2. **Neuron Selection & Filtering**:
   - **Input Layer**: Top 50 most activated neurons (threshold > 0.001)
   - **Hidden Layer**: Top 25 most activated neurons (threshold > 0.001)
   - **Output Layer**: All 10 neurons (digit probabilities)

3. **Activation Mapping**:
   - Maps 2D drawing coordinates to 1D neuron indices
   - Shows which specific pixels/features triggered each neuron
   - Provides normalized activation values for visualization

4. **Real-time Updates**:
   - Extracts activations on every prediction
   - Updates visualization dynamically
   - Shows the neural network's "thinking process"

#### Visualization Features
- **Spatial Mapping**: Input neurons correspond to exact pixel positions
- **Feature Detection**: Hidden neurons show learned pattern responses
- **Decision Process**: Output neurons reveal classification reasoning
- **Activation Intensity**: Color/size coding for neuron activation strength

## 🧪 Testing the Model

### Using the Web Interface
1. **Draw a digit**: Use your mouse or finger to draw on the canvas
2. **Upload an image**: Select an image file containing a digit
3. **View results**: See the prediction, confidence, and probability distribution

### Expected Behavior
- ✅ Clear digits (0-9) should be recognized with high confidence
- ✅ Ambiguous or unclear digits may have lower confidence
- ✅ Non-digit images should trigger error messages
- ✅ Multiple digits or complex images should be rejected

## 🔧 Customization

### Training Parameters
Edit `train_model.py` to modify:
- Number of epochs (current: 200)
- Batch size (current: 32)
- Learning rate (current: 0.002)
- Momentum value (current: 0.95)
- Network architecture

### Network Architecture
Edit `neural_network.py` to change:
- Hidden layer size (current: 64 neurons)
- Activation functions (current: ReLU + Softmax)
- Initialization methods (current: Xavier)
- Optimization algorithms (current: SGD + Momentum)

### Visualization Sensitivity
Edit `app.py` to adjust neuron visualization:
- Number of neurons shown per layer
- Activation thresholds for filtering
- Sensitivity levels for detection

## 📈 Monitoring Training

The training script provides:
- Real-time loss and accuracy updates
- Training history visualization
- Model performance metrics
- Automatic model saving

## 🤝 Contributing

Feel free to improve this project by:
- Adding more activation functions
- Implementing different optimizers
- Enhancing the web interface
- Adding data augmentation
- Improving error handling

## 📝 License

This project is open source and available under the MIT License.

---

**Built with ❤️ using NumPy, Flask, and vanilla JavaScript**
