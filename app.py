from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import base64
import io
from PIL import Image
import os
import pickle
from neural_network import NeuralNetwork
import json

app = Flask(__name__)

# Global variable to store the trained model
model = None

def load_trained_model():
    """Load the trained neural network model"""
    global model
    model_path = "trained_model.pkl"

    if os.path.exists(model_path):
        try:
            # Try loading as direct NeuralNetwork object first (new format)
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)

            if isinstance(model_data, NeuralNetwork):
                # New format: direct NeuralNetwork object
                model = model_data
                print("Model loaded successfully! (Enhanced format)")
                return True
            else:
                # Old format: dictionary with parameters
                model = NeuralNetwork()
                model.load_model(model_path)
                print("Model loaded successfully! (Legacy format)")
                return True

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    else:
        print("No trained model found. Please run train_model.py first.")
        return False

def preprocess_image(image_data):
    """
    Preprocess uploaded image for prediction - optimized for drawn digits
    """
    try:
        # Decode base64 image
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]

        # Convert to PIL Image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))

        # Convert to grayscale if needed
        if image.mode != 'L':
            image = image.convert('L')

        # Convert to numpy array
        img_array = np.array(image)

        # Invert colors if background is white (MNIST has black background)
        if np.mean(img_array) > 127:
            img_array = 255 - img_array

        # Find bounding box of the digit to center it
        coords = np.column_stack(np.where(img_array > 30))  # Find non-background pixels
        if len(coords) > 0:
            # Get bounding box
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)

            # Add some padding
            padding = 20
            y_min = max(0, y_min - padding)
            x_min = max(0, x_min - padding)
            y_max = min(img_array.shape[0], y_max + padding)
            x_max = min(img_array.shape[1], x_max + padding)

            # Crop to bounding box
            img_cropped = img_array[y_min:y_max, x_min:x_max]

            # Resize to 20x20 maintaining aspect ratio
            h, w = img_cropped.shape
            if h > w:
                new_h = 20
                new_w = int(20 * w / h)
            else:
                new_w = 20
                new_h = int(20 * h / w)

            img_resized = cv2.resize(img_cropped, (new_w, new_h))

            # Center in 28x28 image
            img_centered = np.zeros((28, 28), dtype=np.uint8)
            start_y = (28 - new_h) // 2
            start_x = (28 - new_w) // 2
            img_centered[start_y:start_y+new_h, start_x:start_x+new_w] = img_resized

            img_array = img_centered
        else:
            # If no content found, just resize
            img_array = cv2.resize(img_array, (28, 28))

        # Apply slight smoothing
        img_array = cv2.GaussianBlur(img_array, (1, 1), 0)

        # Normalize to [0, 1]
        img_array = img_array.astype(np.float32) / 255.0

        # Flatten to 784 features
        img_flattened = img_array.flatten().reshape(1, -1)

        return img_flattened, img_array

    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None, None

def is_digit_image(img_array, threshold=0.02):
    """
    Very lenient heuristic to check if image contains a digit
    """
    # Check if there's enough non-zero pixels (digit content)
    non_zero_ratio = np.count_nonzero(img_array) / img_array.size

    # Check if the image has some structure (not just noise)
    # Calculate variance to detect if there's actual content
    variance = np.var(img_array)

    # Much more lenient thresholds - let's be less strict!
    has_content = non_zero_ratio > threshold  # Lowered from 0.05 to 0.02
    has_structure = variance > 0.001          # Lowered from 0.005 to 0.001

    print(f"Debug: non_zero_ratio={non_zero_ratio:.3f}, variance={variance:.3f}, has_content={has_content}, has_structure={has_structure}")

    return has_content and has_structure

def get_network_activations(processed_image):
    """
    Extract neural network layer activations for visualization
    """
    try:
        # Forward pass through the network to get activations
        # Input layer (784 neurons)
        input_layer = processed_image.flatten()

        # Hidden layer (forward pass)
        z1 = np.dot(processed_image, model.W1) + model.b1
        hidden_layer = model.relu(z1).flatten()

        # Output layer (forward pass)
        z2 = np.dot(hidden_layer.reshape(1, -1), model.W2) + model.b2
        output_layer = model.softmax(z2).flatten()

        # Get top activated neurons from each layer (more neurons, lower threshold)
        input_top = get_top_neurons(input_layer, 50)  # Show more input neurons
        hidden_top = get_top_neurons(hidden_layer, 25) # Show more hidden neurons
        output_top = get_top_neurons(output_layer, 10)

        return {
            'input_layer': {
                'size': len(input_layer),
                'top_neurons': input_top,
                'activation_summary': {
                    'mean': float(np.mean(input_layer)),
                    'max': float(np.max(input_layer)),
                    'active_count': int(np.sum(input_layer > 0.01))  # Lower threshold for counting active neurons
                }
            },
            'hidden_layer': {
                'size': len(hidden_layer),
                'top_neurons': hidden_top,
                'activation_summary': {
                    'mean': float(np.mean(hidden_layer)),
                    'max': float(np.max(hidden_layer)),
                    'active_count': int(np.sum(hidden_layer > 0.01))  # Lower threshold for counting active neurons
                }
            },
            'output_layer': {
                'size': len(output_layer),
                'neurons': [{'index': i, 'activation': float(val), 'label': str(i)}
                           for i, val in enumerate(output_layer)],
                'activation_summary': {
                    'mean': float(np.mean(output_layer)),
                    'max': float(np.max(output_layer)),
                    'predicted_class': int(np.argmax(output_layer))
                }
            }
        }
    except Exception as e:
        print(f"Error extracting activations: {e}")
        return None

def get_top_neurons(layer_activations, top_k=10):
    """
    Get top-k activated neurons from a layer
    """
    # Get indices of top-k neurons
    top_indices = np.argsort(layer_activations)[-top_k:][::-1]

    return [
        {
            'index': int(idx),
            'activation': float(layer_activations[idx]),
            'normalized': float(layer_activations[idx] / np.max(layer_activations)) if np.max(layer_activations) > 0 else 0
        }
        for idx in top_indices if layer_activations[idx] > 0.001  # Include more neurons (lowered threshold)
    ]

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        if model is None:
            return jsonify({
                'error': 'Model not loaded. Please train the model first.',
                'success': False
            })
        
        # Get image data from request
        data = request.get_json()
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({
                'error': 'No image data provided',
                'success': False
            })
        
        # Preprocess image
        processed_image, img_array = preprocess_image(image_data)
        
        if processed_image is None:
            return jsonify({
                'error': 'Failed to process image',
                'success': False
            })
        
        # Check if image contains a digit (more lenient for drawings)
        if not is_digit_image(img_array):
            return jsonify({
                'error': 'This image does not appear to contain a digit. Please draw or upload a clear digit (0-9).',
                'success': False,
                'is_digit': False,
                'debug_info': {
                    'non_zero_ratio': float(np.count_nonzero(img_array) / img_array.size),
                    'variance': float(np.var(img_array)),
                    'mean_value': float(np.mean(img_array))
                }
            })
        
        # Make prediction
        prediction = model.predict(processed_image)[0]
        probabilities = model.predict_proba(processed_image)[0]
        confidence = float(np.max(probabilities))

        # Get neural network activations for visualization
        activations = get_network_activations(processed_image)

        # Prepare probability distribution
        prob_dist = {str(i): float(probabilities[i]) for i in range(10)}

        return jsonify({
            'success': True,
            'prediction': int(prediction),
            'confidence': confidence,
            'probabilities': prob_dist,
            'activations': activations,
            'is_digit': True
        })
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({
            'error': f'Prediction failed: {str(e)}',
            'success': False
        })

@app.route('/model_info')
def model_info():
    """Get information about the loaded model"""
    if model is None:
        return jsonify({
            'loaded': False,
            'message': 'No model loaded'
        })
    
    return jsonify({
        'loaded': True,
        'architecture': f"{model.input_size} -> {model.hidden_size} -> {model.output_size}",
        'learning_rate': model.learning_rate,
        'momentum': model.momentum,
        'training_epochs': len(model.loss_history) if model.loss_history else 0,
        'final_loss': model.loss_history[-1] if model.loss_history else None
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    print("Starting Neural Network Web Application...")
    print("=" * 50)
    
    # Try to load the trained model
    model_loaded = load_trained_model()
    
    if not model_loaded:
        print("\nWARNING: No trained model found!")
        print("Please run 'python train_model.py' first to train the model.")
        print("The web app will start but predictions won't work until a model is trained.")
    
    print("\nStarting Flask server...")
    print("Open your browser and go to: http://localhost:5000")
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
