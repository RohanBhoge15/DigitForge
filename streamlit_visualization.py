import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
import cv2
from PIL import Image
import io
import pickle
import os
from neural_network import NeuralNetwork
import time

# Set page config
st.set_page_config(
    page_title="Neural Network Visualization",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 0rem;
    }
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 18px;
        font-weight: bold;
    }
    .formula-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 10px 0;
        font-family: monospace;
    }
    .step-title {
        color: #667eea;
        font-size: 20px;
        font-weight: bold;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Load trained model
@st.cache_resource
def load_model():
    model_path = "trained_model.pkl"
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            if isinstance(model_data, NeuralNetwork):
                return model_data
            else:
                model = NeuralNetwork()
                model.load_model(model_path)
                return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    else:
        st.warning("No trained model found. Please run train_model.py first.")
        return None

# Preprocess image
def preprocess_image(image_array):
    """Convert image to 28x28 and normalize"""
    if len(image_array.shape) == 3:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    
    # Invert if needed
    if np.mean(image_array) > 127:
        image_array = 255 - image_array
    
    # Resize to 28x28
    image_resized = cv2.resize(image_array, (28, 28))
    
    # Normalize
    image_normalized = image_resized.astype(np.float32) / 255.0
    
    return image_normalized

# Create visualization for 2D to 1D conversion
def visualize_2d_to_1d_conversion(image_2d):
    """Visualize how 2D image becomes 1D vector"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('#f0f2f6')
    
    # Original 2D image
    ax = axes[0]
    im = ax.imshow(image_2d, cmap='gray', vmin=0, vmax=1)
    ax.set_title('Step 1: Original 2D Image\n(28×28 pixels)', fontsize=14, fontweight='bold', color='#667eea')
    ax.set_xlabel('Width (28)', fontsize=11)
    ax.set_ylabel('Height (28)', fontsize=11)
    plt.colorbar(im, ax=ax, label='Pixel Value')
    
    # Flattening process
    ax = axes[1]
    ax.axis('off')
    ax.text(0.5, 0.9, 'Flattening Process', ha='center', fontsize=14, fontweight='bold', 
            transform=ax.transAxes, color='#667eea')
    
    # Draw arrow and formula
    ax.annotate('', xy=(0.5, 0.7), xytext=(0.5, 0.8),
                arrowprops=dict(arrowstyle='->', lw=3, color='#667eea'),
                xycoords='axes fraction')
    
    formula_text = 'Flatten 28x28 image to 784 features\nShape: (28, 28) → (784,)\nx[i] = image[floor(i/28), i mod 28]'
    
    ax.text(0.5, 0.5, formula_text, ha='center', va='center', fontsize=12,
            transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='#e6e9ff', alpha=0.8),
            family='monospace')
    
    # 1D vector representation
    ax = axes[2]
    flattened = image_2d.flatten()
    ax.bar(range(len(flattened)), flattened, color='#667eea', alpha=0.7, edgecolor='#667eea')
    ax.set_title('Step 2: Flattened 1D Vector\n(784 features)', fontsize=14, fontweight='bold', color='#667eea')
    ax.set_xlabel('Feature Index', fontsize=11)
    ax.set_ylabel('Pixel Value', fontsize=11)
    ax.set_xlim(0, 784)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig

# Create visualization for forward propagation
def visualize_forward_propagation(model, image_1d):
    """Visualize forward propagation step by step"""
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor('#f0f2f6')
    gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)
    
    # Step 1: Input Layer
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(range(len(image_1d)), image_1d, color='#667eea', alpha=0.7, edgecolor='#667eea')
    ax1.set_title('Step 1: Input Layer (784 neurons)', fontsize=12, fontweight='bold', color='#667eea')
    ax1.set_xlabel('Neuron Index')
    ax1.set_ylabel('Activation Value')
    ax1.set_xlim(0, 784)
    ax1.grid(axis='y', alpha=0.3)
    
    # Step 2: Linear transformation (z1 = X @ W1 + b1)
    ax2 = fig.add_subplot(gs[0, 1])
    z1 = np.dot(image_1d.reshape(1, -1), model.W1) + model.b1
    z1_flat = z1.flatten()
    ax2.bar(range(len(z1_flat)), z1_flat, color='#f093fb', alpha=0.7, edgecolor='#f093fb')
    ax2.set_title('Step 2: Linear Transformation\n$z^{(1)} = X \\cdot W^{(1)} + b^{(1)}$', 
                  fontsize=12, fontweight='bold', color='#f093fb')
    ax2.set_xlabel('Hidden Neuron Index')
    ax2.set_ylabel('z Value')
    ax2.grid(axis='y', alpha=0.3)
    
    # Step 3: ReLU Activation
    ax3 = fig.add_subplot(gs[1, 0])
    a1 = model.relu(z1)
    a1_flat = a1.flatten()
    ax3.bar(range(len(a1_flat)), a1_flat, color='#4facfe', alpha=0.7, edgecolor='#4facfe')
    ax3.set_title('Step 3: ReLU Activation\n$a^{(1)} = \\max(0, z^{(1)})$', 
                  fontsize=12, fontweight='bold', color='#4facfe')
    ax3.set_xlabel('Hidden Neuron Index')
    ax3.set_ylabel('Activation Value')
    ax3.grid(axis='y', alpha=0.3)
    
    # Step 4: Output linear transformation
    ax4 = fig.add_subplot(gs[1, 1])
    z2 = np.dot(a1, model.W2) + model.b2
    z2_flat = z2.flatten()
    ax4.bar(range(len(z2_flat)), z2_flat, color='#43e97b', alpha=0.7, edgecolor='#43e97b')
    ax4.set_title('Step 4: Output Linear Transformation\n$z^{(2)} = a^{(1)} \\cdot W^{(2)} + b^{(2)}$', 
                  fontsize=12, fontweight='bold', color='#43e97b')
    ax4.set_xlabel('Output Neuron (Digit)')
    ax4.set_ylabel('z Value')
    ax4.set_xticks(range(10))
    ax4.grid(axis='y', alpha=0.3)
    
    # Step 5: Softmax Probabilities
    ax5 = fig.add_subplot(gs[2, 0])
    a2 = model.softmax(z2)
    a2_flat = a2.flatten()
    colors = ['#fa709a' if i == np.argmax(a2_flat) else '#fa709a' for i in range(10)]
    bars = ax5.bar(range(10), a2_flat, color=colors, alpha=0.7, edgecolor='#fa709a')
    ax5.set_title('Step 5: Softmax Probabilities\n$a^{(2)} = \\text{softmax}(z^{(2)})$', 
                  fontsize=12, fontweight='bold', color='#fa709a')
    ax5.set_xlabel('Digit Class')
    ax5.set_ylabel('Probability')
    ax5.set_xticks(range(10))
    ax5.set_ylim(0, 1)
    ax5.grid(axis='y', alpha=0.3)
    
    # Highlight predicted class
    predicted = np.argmax(a2_flat)
    bars[predicted].set_color('#ff6b6b')
    bars[predicted].set_edgecolor('#ff0000')
    bars[predicted].set_linewidth(2)
    
    # Step 6: Network Architecture Diagram
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.set_xlim(0, 10)
    ax6.set_ylim(0, 10)
    ax6.axis('off')
    
    # Draw network architecture
    ax6.text(5, 9.5, 'Network Architecture', ha='center', fontsize=12, fontweight='bold', color='#667eea')
    
    # Input layer
    ax6.add_patch(FancyBboxPatch((0.5, 7), 1.5, 1.5, boxstyle="round,pad=0.1", 
                                 edgecolor='#667eea', facecolor='#e6e9ff', linewidth=2))
    ax6.text(1.25, 7.75, 'Input\n784', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Hidden layer
    ax6.add_patch(FancyBboxPatch((4, 7), 1.5, 1.5, boxstyle="round,pad=0.1", 
                                 edgecolor='#4facfe', facecolor='#e0f7ff', linewidth=2))
    ax6.text(4.75, 7.75, 'Hidden\n64', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Output layer
    ax6.add_patch(FancyBboxPatch((7.5, 7), 1.5, 1.5, boxstyle="round,pad=0.1", 
                                 edgecolor='#fa709a', facecolor='#ffe0e6', linewidth=2))
    ax6.text(8.25, 7.75, 'Output\n10', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Arrows
    ax6.annotate('', xy=(4, 7.75), xytext=(2, 7.75),
                arrowprops=dict(arrowstyle='->', lw=2, color='#667eea'))
    ax6.annotate('', xy=(7.5, 7.75), xytext=(5.5, 7.75),
                arrowprops=dict(arrowstyle='->', lw=2, color='#4facfe'))
    
    # Add formulas
    ax6.text(3, 6.2, r'$z^{(1)} = X \cdot W^{(1)} + b^{(1)}$', ha='center', fontsize=9, 
            bbox=dict(boxstyle='round', facecolor='#fff9e6', alpha=0.8))
    ax6.text(3, 5.7, r'$a^{(1)} = ReLU(z^{(1)})$', ha='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='#fff9e6', alpha=0.8))
    
    ax6.text(6.25, 6.2, r'$z^{(2)} = a^{(1)} \cdot W^{(2)} + b^{(2)}$', ha='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='#fff9e6', alpha=0.8))
    ax6.text(6.25, 5.7, r'$a^{(2)} = softmax(z^{(2)})$', ha='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='#fff9e6', alpha=0.8))
    
    # Prediction result
    ax6.text(5, 4.5, f'Predicted Digit: {predicted}', ha='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#ff6b6b', alpha=0.3, edgecolor='#ff6b6b', linewidth=2))
    ax6.text(5, 3.8, f'Confidence: {a2_flat[predicted]:.2%}', ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='#90ee90', alpha=0.3))
    
    plt.tight_layout()
    return fig

# Create detailed formula visualization
def create_formula_visualization():
    """Create detailed formulas used in forward propagation"""
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor('#f0f2f6')
    ax.axis('off')
    
    y_pos = 0.95
    
    # Title
    ax.text(0.5, y_pos, 'Forward Propagation Formulas', ha='center', fontsize=18, 
            fontweight='bold', color='#667eea', transform=ax.transAxes)
    y_pos -= 0.08
    
    # Input Layer
    ax.text(0.05, y_pos, '1. INPUT LAYER', fontsize=13, fontweight='bold', 
            color='#667eea', transform=ax.transAxes)
    y_pos -= 0.06
    ax.text(0.1, y_pos, 'x in R^784 - Flattened 28x28 image pixels', 
            fontsize=11, transform=ax.transAxes, family='monospace',
            bbox=dict(boxstyle='round', facecolor='#e6e9ff', alpha=0.8))
    y_pos -= 0.08
    
    # Linear Transformation 1
    ax.text(0.05, y_pos, '2. HIDDEN LAYER - LINEAR TRANSFORMATION', fontsize=13, 
            fontweight='bold', color='#f093fb', transform=ax.transAxes)
    y_pos -= 0.06
    ax.text(0.1, y_pos, 'z(1) = x * W(1) + b(1)', 
            fontsize=12, transform=ax.transAxes, family='monospace',
            bbox=dict(boxstyle='round', facecolor='#ffe0f0', alpha=0.8))
    y_pos -= 0.06
    ax.text(0.1, y_pos, 'where: W(1) in R^(784x64), b(1) in R^64', 
            fontsize=10, transform=ax.transAxes, style='italic', color='#555')
    y_pos -= 0.08
    
    # ReLU Activation
    ax.text(0.05, y_pos, '3. HIDDEN LAYER - ACTIVATION (ReLU)', fontsize=13, 
            fontweight='bold', color='#4facfe', transform=ax.transAxes)
    y_pos -= 0.06
    ax.text(0.1, y_pos, 'a(1) = ReLU(z(1)) = max(0, z(1))', 
            fontsize=12, transform=ax.transAxes, family='monospace',
            bbox=dict(boxstyle='round', facecolor='#e0f7ff', alpha=0.8))
    y_pos -= 0.06
    ax.text(0.1, y_pos, 'Element-wise: a(1)_i = z(1)_i if z(1)_i > 0, else 0', 
            fontsize=10, transform=ax.transAxes, style='italic', color='#555')
    y_pos -= 0.08
    
    # Linear Transformation 2
    ax.text(0.05, y_pos, '4. OUTPUT LAYER - LINEAR TRANSFORMATION', fontsize=13, 
            fontweight='bold', color='#43e97b', transform=ax.transAxes)
    y_pos -= 0.06
    ax.text(0.1, y_pos, 'z(2) = a(1) * W(2) + b(2)', 
            fontsize=12, transform=ax.transAxes, family='monospace',
            bbox=dict(boxstyle='round', facecolor='#e0ffe0', alpha=0.8))
    y_pos -= 0.06
    ax.text(0.1, y_pos, 'where: W(2) in R^(64x10), b(2) in R^10', 
            fontsize=10, transform=ax.transAxes, style='italic', color='#555')
    y_pos -= 0.08
    
    # Softmax
    ax.text(0.05, y_pos, '5. OUTPUT LAYER - SOFTMAX ACTIVATION', fontsize=13, 
            fontweight='bold', color='#fa709a', transform=ax.transAxes)
    y_pos -= 0.06
    ax.text(0.1, y_pos, 'a(2)_j = exp(z(2)_j) / sum(exp(z(2)_k)) for k=0 to 9', 
            fontsize=12, transform=ax.transAxes, family='monospace',
            bbox=dict(boxstyle='round', facecolor='#ffe0e6', alpha=0.8))
    y_pos -= 0.06
    ax.text(0.1, y_pos, 'Produces probability distribution: sum(a(2)_j) = 1 for j=0 to 9', 
            fontsize=10, transform=ax.transAxes, style='italic', color='#555')
    y_pos -= 0.08
    
    # Prediction
    ax.text(0.05, y_pos, '6. PREDICTION', fontsize=13, fontweight='bold', 
            color='#ff6b6b', transform=ax.transAxes)
    y_pos -= 0.06
    ax.text(0.1, y_pos, 'y_hat = argmax(a(2)_j) for j=0 to 9', 
            fontsize=12, transform=ax.transAxes, family='monospace',
            bbox=dict(boxstyle='round', facecolor='#ffe0e0', alpha=0.8))
    y_pos -= 0.06
    ax.text(0.1, y_pos, 'Confidence: conf = max(a(2))', 
            fontsize=10, transform=ax.transAxes, style='italic', color='#555')
    
    plt.tight_layout()
    return fig

# Create weight visualization
def visualize_weights(model):
    """Visualize weight matrices"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#f0f2f6')
    
    # W1 visualization
    ax = axes[0]
    im1 = ax.imshow(model.W1, cmap='RdBu_r', aspect='auto')
    ax.set_title('Weight Matrix W¹ (784×64)\nInput → Hidden Layer', fontsize=12, fontweight='bold', color='#667eea')
    ax.set_xlabel('Hidden Neurons (64)')
    ax.set_ylabel('Input Features (784)')
    plt.colorbar(im1, ax=ax, label='Weight Value')
    
    # W2 visualization
    ax = axes[1]
    im2 = ax.imshow(model.W2, cmap='RdBu_r', aspect='auto')
    ax.set_title('Weight Matrix W² (64×10)\nHidden → Output Layer', fontsize=12, fontweight='bold', color='#667eea')
    ax.set_xlabel('Output Classes (10)')
    ax.set_ylabel('Hidden Neurons (64)')
    plt.colorbar(im2, ax=ax, label='Weight Value')
    
    plt.tight_layout()
    return fig

# Create activation heatmap
def visualize_activations(model, image_1d):
    """Visualize activations at each layer"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.patch.set_facecolor('#f0f2f6')
    
    # Input activations (as 28x28 image)
    ax = axes[0]
    input_2d = image_1d.reshape(28, 28)
    im1 = ax.imshow(input_2d, cmap='hot')
    ax.set_title('Input Layer Activations\n(28×28 Image)', fontsize=12, fontweight='bold', color='#667eea')
    plt.colorbar(im1, ax=ax, label='Activation')
    
    # Hidden layer activations
    ax = axes[1]
    z1 = np.dot(image_1d.reshape(1, -1), model.W1) + model.b1
    a1 = model.relu(z1).flatten()
    ax.bar(range(len(a1)), a1, color='#4facfe', alpha=0.7, edgecolor='#4facfe')
    ax.set_title('Hidden Layer Activations\n(64 Neurons)', fontsize=12, fontweight='bold', color='#667eea')
    ax.set_xlabel('Neuron Index')
    ax.set_ylabel('Activation Value')
    ax.grid(axis='y', alpha=0.3)
    
    # Output layer activations (probabilities)
    ax = axes[2]
    z2 = np.dot(a1.reshape(1, -1), model.W2) + model.b2
    a2 = model.softmax(z2).flatten()
    colors = ['#ff6b6b' if i == np.argmax(a2) else '#fa709a' for i in range(10)]
    ax.bar(range(10), a2, color=colors, alpha=0.7, edgecolor='#fa709a')
    ax.set_title('Output Layer Probabilities\n(10 Classes)', fontsize=12, fontweight='bold', color='#667eea')
    ax.set_xlabel('Digit Class')
    ax.set_ylabel('Probability')
    ax.set_xticks(range(10))
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig

# Main Streamlit App
def main():
    st.title("🧠 Neural Network Forward Propagation Visualization")
    st.markdown("---")
    
    # Load model
    model = load_model()
    if model is None:
        st.error("Cannot proceed without a trained model.")
        return
    
    # Sidebar for input
    st.sidebar.header("📊 Input Options")
    
    input_method = st.sidebar.radio("Choose input method:", ["Draw Digit", "Upload Image"])
    
    image_input = None
    
    if input_method == "Draw Digit":
        st.sidebar.markdown("### Draw a digit on the canvas below")
        
        # Create a simple drawing interface using streamlit-drawable-canvas
        try:
            from streamlit_drawable_canvas import st_canvas
            
            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 1)",
                stroke_width=15,
                stroke_color="rgba(0, 0, 0, 1)",
                background_color="rgba(255, 255, 255, 1)",
                height=280,
                width=280,
                drawing_mode="freedraw",
                key="canvas",
            )
            
            if canvas_result.image_data is not None:
                image_input = canvas_result.image_data
        except ImportError:
            st.warning("Please install streamlit-drawable-canvas: pip install streamlit-drawable-canvas")
            
            # Fallback: use PIL Image
            st.info("Using PIL Image upload as fallback")
            uploaded_file = st.file_uploader("Upload a digit image", type=['png', 'jpg', 'jpeg'])
            if uploaded_file:
                image_input = Image.open(uploaded_file)
                image_input = np.array(image_input)
    
    else:  # Upload Image
        uploaded_file = st.file_uploader("Upload a digit image", type=['png', 'jpg', 'jpeg'])
        if uploaded_file:
            image_input = Image.open(uploaded_file)
            image_input = np.array(image_input)
    
    # Process and visualize
    if image_input is not None:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("📸 Input Image")
            st.image(image_input, use_column_width=True)
        
        with col2:
            st.subheader("🔄 Preprocessing")
            image_processed = preprocess_image(image_input)
            st.image(image_processed, use_column_width=True, clamp=True)
        
        # Show simulation button
        if st.button("🎬 Show Detailed Simulation", key="simulate", use_container_width=True):
            st.markdown("---")
            
            # Tab 1: 2D to 1D Conversion
            st.subheader("📐 Step 1: 2D Image to 1D Vector Conversion")
            st.markdown("""
            The 28×28 pixel image is flattened into a 1D vector of 784 features.
            This is the input to our neural network.
            """)
            
            fig_2d_1d = visualize_2d_to_1d_conversion(image_processed)
            st.pyplot(fig_2d_1d, use_container_width=True)
            
            with st.expander("📝 Formula Details"):
                st.markdown("""
                **Flattening Formula:**
                ```
                x = image.flatten()
                Shape: (28, 28) → (784,)
                x[i] = image[floor(i/28), i mod 28]
                ```
                """)
            
            st.markdown("---")
            
            # Tab 2: Forward Propagation
            st.subheader("🔄 Step 2: Forward Propagation Through Network")
            st.markdown("""
            The input vector passes through the network layers:
            1. **Input Layer**: 784 neurons (one per pixel)
            2. **Hidden Layer**: 64 neurons with ReLU activation
            3. **Output Layer**: 10 neurons with Softmax activation
            """)
            
            fig_forward = visualize_forward_propagation(model, image_processed.flatten())
            st.pyplot(fig_forward, use_container_width=True)
            
            st.markdown("---")
            
            # Tab 3: Detailed Formulas
            st.subheader("📐 Step 3: Mathematical Formulas")
            
            fig_formulas = create_formula_visualization()
            st.pyplot(fig_formulas, use_container_width=True)
            
            st.markdown("---")
            
            # Tab 4: Weight Matrices
            st.subheader("⚖️ Step 4: Weight Matrices Visualization")
            st.markdown("""
            These are the learned parameters that transform data between layers.
            Darker colors represent negative weights, lighter colors represent positive weights.
            """)
            
            fig_weights = visualize_weights(model)
            st.pyplot(fig_weights, use_container_width=True)
            
            st.markdown("---")
            
            # Tab 5: Activations
            st.subheader("🔥 Step 5: Layer Activations")
            st.markdown("""
            Visualization of how the input is transformed through each layer.
            """)
            
            fig_activations = visualize_activations(model, image_processed.flatten())
            st.pyplot(fig_activations, use_container_width=True)
            
            st.markdown("---")
            
            # Tab 6: Detailed Breakdown
            st.subheader("📊 Step 6: Detailed Numerical Breakdown")
            
            image_1d = image_processed.flatten()
            
            # Layer 1
            z1 = np.dot(image_1d.reshape(1, -1), model.W1) + model.b1
            a1 = model.relu(z1)
            
            # Layer 2
            z2 = np.dot(a1, model.W2) + model.b2
            a2 = model.softmax(z2)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### 🔢 Input Layer")
                st.write(f"Shape: {image_1d.shape}")
                st.write(f"Min: {image_1d.min():.4f}")
                st.write(f"Max: {image_1d.max():.4f}")
                st.write(f"Mean: {image_1d.mean():.4f}")
                st.write(f"Non-zero: {np.count_nonzero(image_1d)}")
            
            with col2:
                st.markdown("### 🧠 Hidden Layer")
                st.write(f"Shape: {a1.shape}")
                st.write(f"Min: {a1.min():.4f}")
                st.write(f"Max: {a1.max():.4f}")
                st.write(f"Mean: {a1.mean():.4f}")
                st.write(f"Active: {np.count_nonzero(a1)}")
            
            with col3:
                st.markdown("### 📤 Output Layer")
                st.write(f"Shape: {a2.shape}")
                st.write(f"Min: {a2.min():.4f}")
                st.write(f"Max: {a2.max():.4f}")
                st.write(f"Mean: {a2.mean():.4f}")
                st.write(f"Sum: {a2.sum():.4f}")
            
            st.markdown("---")
            
            # Final Prediction
            st.subheader("🎯 Final Prediction")
            
            predicted_digit = np.argmax(a2.flatten())
            confidence = a2.flatten()[predicted_digit]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Predicted Digit", predicted_digit, delta=None)
            
            with col2:
                st.metric("Confidence", f"{confidence:.2%}", delta=None)
            
            with col3:
                st.metric("Entropy", f"{-np.sum(a2 * np.log(a2 + 1e-10)):.4f}", delta=None)
            
            # Probability distribution
            st.markdown("### 📊 Probability Distribution")
            
            probs = a2.flatten()
            fig, ax = plt.subplots(figsize=(12, 5))
            fig.patch.set_facecolor('#f0f2f6')
            
            colors = ['#ff6b6b' if i == predicted_digit else '#667eea' for i in range(10)]
            bars = ax.bar(range(10), probs, color=colors, alpha=0.7, edgecolor='#667eea', linewidth=2)
            
            ax.set_title('Output Layer Probabilities (Softmax)', fontsize=14, fontweight='bold', color='#667eea')
            ax.set_xlabel('Digit Class', fontsize=12)
            ax.set_ylabel('Probability', fontsize=12)
            ax.set_xticks(range(10))
            ax.set_ylim(0, 1)
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for i, (bar, prob) in enumerate(zip(bars, probs)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{prob:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Detailed step-by-step explanation
            st.subheader("📖 Step-by-Step Explanation")
            
            with st.expander("🔍 Expand for detailed mathematical breakdown"):
                st.markdown(f"""
                ### Forward Propagation Calculation
                
                **Input Vector (x):**
                - Shape: (784,)
                - Values: Normalized pixel intensities [0, 1]
                
                **Step 1: Hidden Layer Linear Transformation**
                ```
                z¹ = x · W¹ + b¹
                z¹ ∈ ℝ⁶⁴
                ```
                - Matrix multiplication: (784,) · (784, 64) = (64,)
                - Add bias: (64,) + (64,) = (64,)
                
                **Step 2: Hidden Layer Activation (ReLU)**
                ```
                a¹ = max(0, z¹)
                a¹ ∈ ℝ⁶⁴
                ```
                - Element-wise: a¹ᵢ = max(0, z¹ᵢ)
                - Introduces non-linearity
                - Sparsity: {np.count_nonzero(a1)}/{len(a1.flatten())} neurons active
                
                **Step 3: Output Layer Linear Transformation**
                ```
                z² = a¹ · W² + b²
                z² ∈ ℝ¹⁰
                ```
                - Matrix multiplication: (64,) · (64, 10) = (10,)
                - Add bias: (10,) + (10,) = (10,)
                
                **Step 4: Output Layer Activation (Softmax)**
                ```
                a²ⱼ = exp(z²ⱼ) / Σₖ exp(z²ₖ)
                a² ∈ ℝ¹⁰, Σⱼ a²ⱼ = 1
                ```
                - Converts logits to probabilities
                - All values in [0, 1]
                - Sum equals 1
                
                **Step 5: Prediction**
                ```
                ŷ = argmax(a²)
                confidence = max(a²)
                ```
                - Predicted digit: {predicted_digit}
                - Confidence: {confidence:.4f}
                """)
            
            st.success("✅ Simulation Complete!")

if __name__ == "__main__":
    main()
