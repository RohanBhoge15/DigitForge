// Global variables
let canvas, ctx;
let isDrawing = false;
let lastX = 0;
let lastY = 0;

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeCanvas();
    initializeEventListeners();
    loadModelInfo();
});

// Initialize drawing canvas
function initializeCanvas() {
    canvas = document.getElementById('drawingCanvas');
    ctx = canvas.getContext('2d');
    
    // Set canvas properties
    ctx.strokeStyle = '#000';
    ctx.lineWidth = 15;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    
    // Clear canvas with white background
    clearCanvas();
}

// Initialize event listeners
function initializeEventListeners() {
    // Canvas drawing events
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);
    
    // Touch events for mobile
    canvas.addEventListener('touchstart', handleTouch);
    canvas.addEventListener('touchmove', handleTouch);
    canvas.addEventListener('touchend', stopDrawing);
    
    // Button events
    document.getElementById('clearCanvas').addEventListener('click', clearCanvas);
    document.getElementById('predictDrawing').addEventListener('click', predictDrawing);
    document.getElementById('predictUpload').addEventListener('click', predictUpload);
    document.getElementById('compareAll').addEventListener('click', compareAllModels);

    // File upload events
    document.getElementById('imageUpload').addEventListener('change', handleFileUpload);

    // Model selection events
    document.querySelectorAll('input[name="modelType"]').forEach(radio => {
        radio.addEventListener('change', handleModelTypeChange);
    });
}

// Canvas drawing functions
function startDrawing(e) {
    isDrawing = true;
    [lastX, lastY] = getMousePos(e);
}

function draw(e) {
    if (!isDrawing) return;
    
    const [currentX, currentY] = getMousePos(e);
    
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(currentX, currentY);
    ctx.stroke();
    
    [lastX, lastY] = [currentX, currentY];
}

function stopDrawing() {
    isDrawing = false;
}

function getMousePos(e) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    return [
        (e.clientX - rect.left) * scaleX,
        (e.clientY - rect.top) * scaleY
    ];
}

// Touch event handling
function handleTouch(e) {
    e.preventDefault();
    const touch = e.touches[0];
    const mouseEvent = new MouseEvent(e.type === 'touchstart' ? 'mousedown' : 
                                     e.type === 'touchmove' ? 'mousemove' : 'mouseup', {
        clientX: touch.clientX,
        clientY: touch.clientY
    });
    canvas.dispatchEvent(mouseEvent);
}

// Clear canvas
function clearCanvas() {
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    hideResults();
}

// File upload handling
function handleFileUpload(e) {
    const file = e.target.files[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = function(event) {
        const img = document.getElementById('imagePreview');
        img.src = event.target.result;
        img.style.display = 'block';
        document.getElementById('predictUpload').style.display = 'block';
        hideResults();
    };
    reader.readAsDataURL(file);
}

// Model selection functions
function handleModelTypeChange() {
    const mlSelect = document.getElementById('mlModelSelect');
    const selectedType = document.querySelector('input[name="modelType"]:checked').value;

    if (selectedType === 'ml') {
        mlSelect.disabled = false;
    } else {
        mlSelect.disabled = true;
    }
}

// Prediction functions
async function predictDrawing() {
    const imageData = canvas.toDataURL('image/png');
    await makePrediction(imageData);
}

async function predictUpload() {
    const img = document.getElementById('imagePreview');
    if (!img.src) {
        showError('No image selected');
        return;
    }
    
    // Convert image to canvas to get consistent format
    const tempCanvas = document.createElement('canvas');
    const tempCtx = tempCanvas.getContext('2d');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    
    const image = new Image();
    image.onload = async function() {
        tempCtx.drawImage(image, 0, 0, 28, 28);
        const imageData = tempCanvas.toDataURL('image/png');
        await makePrediction(imageData);
    };
    image.src = img.src;
}

// Make prediction API call
async function makePrediction(imageData) {
    showLoading();
    hideComparison();

    const selectedType = document.querySelector('input[name="modelType"]:checked').value;
    let endpoint = '/predict';
    let requestBody = { image: imageData };

    if (selectedType === 'ml') {
        endpoint = '/predict_ml';
        requestBody.model = document.getElementById('mlModelSelect').value;
    }

    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestBody)
        });

        const result = await response.json();
        hideLoading();

        if (result.success) {
            showResults(result);
            if (result.activations && selectedType === 'custom') {
                showNetworkVisualization(result.activations);
            }
        } else {
            showError(result.error);
        }

    } catch (error) {
        hideLoading();
        showError('Network error: ' + error.message);
    }
}

// Compare all models
async function compareAllModels() {
    const imageData = canvas.toDataURL('image/png');
    showLoading();
    hideResults();

    try {
        const response = await fetch('/compare_models', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: imageData })
        });

        const result = await response.json();
        hideLoading();

        if (result.success) {
            showComparison(result.results);
        } else {
            showError(result.error);
        }

    } catch (error) {
        hideLoading();
        showError('Network error: ' + error.message);
    }
}

// Display results
function showResults(result) {
    hideError();
    hideComparison();

    const resultsSection = document.getElementById('resultsSection');
    const predictedDigit = document.getElementById('predictedDigit');
    const confidence = document.getElementById('confidence');
    const modelUsed = document.getElementById('modelUsed');
    const probabilityBars = document.getElementById('probabilityBars');

    // Update prediction display
    predictedDigit.textContent = result.prediction;
    confidence.textContent = `Confidence: ${(result.confidence * 100).toFixed(1)}%`;

    // Update model information
    if (result.model_name) {
        modelUsed.textContent = `Model: ${result.model_name} (${result.model_type || 'Custom'})`;
    } else {
        modelUsed.textContent = 'Model: Custom Neural Network';
    }
    
    // Create probability bars
    probabilityBars.innerHTML = '';
    for (let i = 0; i < 10; i++) {
        const probability = result.probabilities[i.toString()];
        const percentage = (probability * 100).toFixed(1);
        
        const barContainer = document.createElement('div');
        barContainer.className = 'prob-bar';
        
        barContainer.innerHTML = `
            <div class="prob-label">${i}</div>
            <div class="prob-bar-bg">
                <div class="prob-bar-fill" style="width: ${percentage}%"></div>
            </div>
            <div class="prob-percentage">${percentage}%</div>
        `;
        
        probabilityBars.appendChild(barContainer);
    }
    
    resultsSection.style.display = 'block';
}

// Display error
function showError(message) {
    hideResults();
    
    const errorSection = document.getElementById('errorSection');
    const errorMessage = document.getElementById('errorMessage');
    
    errorMessage.textContent = message;
    errorSection.style.display = 'block';
}

// Hide results and errors
function hideResults() {
    document.getElementById('resultsSection').style.display = 'none';
}

function hideError() {
    document.getElementById('errorSection').style.display = 'none';
}

// Loading spinner
function showLoading() {
    document.getElementById('loadingSpinner').style.display = 'flex';
}

function hideLoading() {
    document.getElementById('loadingSpinner').style.display = 'none';
}

// Load model information
async function loadModelInfo() {
    try {
        const response = await fetch('/model_info');
        const info = await response.json();
        
        const modelInfoDiv = document.getElementById('modelInfo');
        
        if (info.loaded) {
            modelInfoDiv.innerHTML = `
                <p><strong>Status:</strong> Model loaded successfully ✅</p>
                <p><strong>Architecture:</strong> ${info.architecture}</p>
                <p><strong>Learning Rate:</strong> ${info.learning_rate}</p>
                <p><strong>Momentum:</strong> ${info.momentum}</p>
                <p><strong>Training Epochs:</strong> ${info.training_epochs}</p>
                ${info.final_loss ? `<p><strong>Final Loss:</strong> ${info.final_loss.toFixed(4)}</p>` : ''}
            `;
        } else {
            modelInfoDiv.innerHTML = `
                <p><strong>Status:</strong> No model loaded ❌</p>
                <p>Please run <code>python train_model.py</code> to train the model first.</p>
            `;
        }
        
    } catch (error) {
        document.getElementById('modelInfo').innerHTML = `
            <p><strong>Status:</strong> Error loading model info ❌</p>
            <p>Error: ${error.message}</p>
        `;
    }
}

// Neural Network Visualization
function showNetworkVisualization(activations) {
    const networkViz = document.getElementById('networkVisualization');
    networkViz.style.display = 'block';

    // Visualize input layer
    visualizeInputLayer(activations.input_layer);

    // Visualize hidden layer
    visualizeHiddenLayer(activations.hidden_layer);

    // Visualize output layer
    visualizeOutputLayer(activations.output_layer);
}

function visualizeInputLayer(inputLayer) {
    const stats = document.getElementById('inputLayerStats');
    const viz = document.getElementById('inputLayerViz');

    // Show statistics
    stats.innerHTML = `
        <strong>Active Neurons:</strong> ${inputLayer.activation_summary.active_count}/${inputLayer.size} |
        <strong>Max Activation:</strong> ${inputLayer.activation_summary.max.toFixed(3)} |
        <strong>Mean:</strong> ${inputLayer.activation_summary.mean.toFixed(3)}
    `;

    // Create 28x28 pixel grid visualization
    viz.innerHTML = '';
    const topNeurons = inputLayer.top_neurons.slice(0, 784); // All pixels

    // Create a simplified visualization showing the most active pixels
    for (let i = 0; i < 784; i++) {
        const pixel = document.createElement('div');
        pixel.className = 'neuron-pixel';

        // Find if this pixel is in the top neurons
        const neuronData = topNeurons.find(n => n.index === i);
        const activation = neuronData ? neuronData.activation : 0;

        // Set pixel intensity based on activation
        const intensity = Math.min(255, Math.floor(activation * 255));
        pixel.style.backgroundColor = `rgb(${intensity}, ${intensity}, ${intensity})`;

        viz.appendChild(pixel);
    }
}

function visualizeHiddenLayer(hiddenLayer) {
    const stats = document.getElementById('hiddenLayerStats');
    const viz = document.getElementById('hiddenLayerViz');

    // Show statistics
    stats.innerHTML = `
        <strong>Active Neurons:</strong> ${hiddenLayer.activation_summary.active_count}/${hiddenLayer.size} |
        <strong>Max Activation:</strong> ${hiddenLayer.activation_summary.max.toFixed(3)} |
        <strong>Mean:</strong> ${hiddenLayer.activation_summary.mean.toFixed(3)}
    `;

    // Show top activated neurons
    viz.innerHTML = '';
    hiddenLayer.top_neurons.forEach(neuron => {
        const neuronDiv = document.createElement('div');
        neuronDiv.className = 'neuron-item';
        neuronDiv.innerHTML = `
            <strong>N${neuron.index}</strong><br>
            ${neuron.activation.toFixed(3)}
        `;

        // Color intensity based on activation
        const intensity = neuron.normalized;
        neuronDiv.style.background = `rgba(102, 126, 234, ${0.1 + intensity * 0.4})`;

        viz.appendChild(neuronDiv);
    });
}

function visualizeOutputLayer(outputLayer) {
    const stats = document.getElementById('outputLayerStats');
    const viz = document.getElementById('outputLayerViz');

    // Show statistics
    stats.innerHTML = `
        <strong>Predicted Class:</strong> ${outputLayer.activation_summary.predicted_class} |
        <strong>Max Probability:</strong> ${outputLayer.activation_summary.max.toFixed(3)} |
        <strong>Mean:</strong> ${outputLayer.activation_summary.mean.toFixed(3)}
    `;

    // Show all output neurons (digits 0-9)
    viz.innerHTML = '';
    outputLayer.neurons.forEach(neuron => {
        const neuronDiv = document.createElement('div');
        neuronDiv.className = 'output-neuron';

        // Highlight the predicted digit
        if (neuron.index === outputLayer.activation_summary.predicted_class) {
            neuronDiv.classList.add('predicted');
        }

        neuronDiv.innerHTML = `
            <div class="digit">${neuron.label}</div>
            <div class="probability">${(neuron.activation * 100).toFixed(1)}%</div>
            <div class="activation-bar">
                <div class="activation-fill" style="width: ${neuron.activation * 100}%"></div>
            </div>
        `;

        viz.appendChild(neuronDiv);
    });
}

// Show comparison results
function showComparison(results) {
    hideError();
    hideResults();

    const comparisonSection = document.getElementById('comparisonSection');
    const comparisonTable = document.getElementById('comparisonTable');

    // Sort results by confidence
    const sortedResults = Object.entries(results)
        .filter(([name, result]) => !result.error)
        .sort(([,a], [,b]) => b.confidence - a.confidence);

    // Create comparison table
    comparisonTable.innerHTML = '<h3>📊 Model Performance Comparison</h3>';

    sortedResults.forEach(([modelName, result], index) => {
        const resultDiv = document.createElement('div');
        resultDiv.className = 'model-result';

        // Highlight best and worst
        if (index === 0) resultDiv.classList.add('best');
        if (index === sortedResults.length - 1 && sortedResults.length > 1) {
            resultDiv.classList.add('worst');
        }

        resultDiv.innerHTML = `
            <div>
                <div class="model-name">${modelName}</div>
                <div class="model-type">${result.model_type || 'Custom'}</div>
            </div>
            <div>
                <div class="model-prediction">Predicted: ${result.prediction}</div>
                <div class="model-confidence">Confidence: ${(result.confidence * 100).toFixed(1)}%</div>
            </div>
        `;

        comparisonTable.appendChild(resultDiv);
    });

    // Show errors if any
    const errors = Object.entries(results).filter(([name, result]) => result.error);
    if (errors.length > 0) {
        const errorDiv = document.createElement('div');
        errorDiv.innerHTML = '<h4>⚠️ Model Errors:</h4>';
        errors.forEach(([name, result]) => {
            errorDiv.innerHTML += `<p><strong>${name}:</strong> ${result.error}</p>`;
        });
        comparisonTable.appendChild(errorDiv);
    }

    comparisonSection.style.display = 'block';
}

// Hide comparison
function hideComparison() {
    document.getElementById('comparisonSection').style.display = 'none';
}

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeCanvas();
    loadModelInfo();
    handleModelTypeChange(); // Initialize model selection
});
