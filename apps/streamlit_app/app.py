"""
Streamlit App for Model Deployment Demo
Lecture 83 - Deployment
"""

import streamlit as st
import numpy as np
import requests
from PIL import Image
import io

st.set_page_config(
    page_title="Fashion-MNIST Classifier",
    page_icon="👔",
    layout="wide"
)

# Configuration
API_URL = "http://localhost:8000"

st.title("👔 Fashion-MNIST Classifier")
st.markdown("Upload an image or draw to classify fashion items!")

# Sidebar
with st.sidebar:
    st.header("Settings")
    api_url = st.text_input("API URL", value=API_URL)
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This app demonstrates model deployment using:
    - **Backend**: FastAPI serving a CNN model
    - **Frontend**: Streamlit for interactive UI
    - **Model**: Fashion-MNIST classifier
    """)

# Check API health
def check_api_health():
    try:
        response = requests.get(f"{api_url}/ping", timeout=2)
        return response.status_code == 200
    except:
        return False

is_healthy = check_api_health()

if is_healthy:
    st.success("✓ API is online")
else:
    st.error("✗ API is offline. Start the FastAPI server first.")
    st.stop()

# Main interface
tab1, tab2, tab3 = st.tabs(["📤 Upload Image", "✏️ Draw", "ℹ️ Model Info"])

with tab1:
    st.header("Upload an Image")
    uploaded_file = st.file_uploader("Choose a 28x28 grayscale image", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert('L')
        image = image.resize((28, 28))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Uploaded Image", width=280)
        
        with col2:
            if st.button("Classify", key="upload"):
                # Convert to array
                img_array = np.array(image).tolist()
                
                # Make prediction
                with st.spinner("Classifying..."):
                    response = requests.post(
                        f"{api_url}/predict",
                        json={"instances": [img_array]}
                    )
                
                if response.status_code == 200:
                    result = response.json()
                    pred = result['predictions'][0]
                    
                    st.success(f"**Prediction: {pred['class_name']}**")
                    st.metric("Confidence", f"{pred['confidence']:.2%}")
                    
                    # Show all probabilities
                    st.subheader("All Class Probabilities")
                    probs = pred['probabilities']
                    metadata_response = requests.get(f"{api_url}/metadata")
                    class_names = metadata_response.json()['class_names']
                    
                    for name, prob in zip(class_names, probs):
                        st.progress(prob, text=f"{name}: {prob:.2%}")
                else:
                    st.error(f"Error: {response.text}")

with tab2:
    st.header("Draw a Fashion Item")
    st.markdown("Use the drawing canvas below (28x28 pixels)")
    
    # Simple drawing interface
    from streamlit_drawable_canvas import st_canvas
    
    canvas_result = st_canvas(
        stroke_width=3,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas"
    )
    
    if st.button("Classify Drawing", key="draw") and canvas_result.image_data is not None:
        # Convert canvas to 28x28 grayscale
        img = Image.fromarray(canvas_result.image_data.astype('uint8')).convert('L')
        img = img.resize((28, 28))
        img_array = np.array(img).tolist()
        
        # Make prediction
        with st.spinner("Classifying..."):
            response = requests.post(
                f"{api_url}/predict",
                json={"instances": [img_array]}
            )
        
        if response.status_code == 200:
            result = response.json()
            pred = result['predictions'][0]
            
            st.success(f"**Prediction: {pred['class_name']}**")
            st.metric("Confidence", f"{pred['confidence']:.2%}")

with tab3:
    st.header("Model Information")
    
    try:
        response = requests.get(f"{api_url}/metadata")
        if response.status_code == 200:
            metadata = response.json()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Model Name", metadata['model_name'])
                st.metric("Version", metadata['version'])
                st.metric("Framework", metadata['framework'])
            
            with col2:
                st.metric("Input Shape", str(metadata['input_shape']))
                st.metric("Output Classes", metadata['output_classes'])
                st.metric("Framework Version", metadata['framework_version'])
            
            st.subheader("Class Names")
            st.write(metadata['class_names'])
    except Exception as e:
        st.error(f"Could not fetch metadata: {e}")

# Footer
st.markdown("---")
st.markdown("**Lecture 83 - Model Deployment** | Built with Streamlit + FastAPI")
