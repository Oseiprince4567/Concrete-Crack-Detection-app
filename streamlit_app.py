import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import os
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Concrete Crack Detection Web App",
    page_icon="🧱",
    layout="wide"
)

# Load model (cache to avoid reloading on every run)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("Crack Detection/crack_model.h5")

model = load_model()

# Prediction function
def predict_crack_pil(img_pil):
    img = img_pil.resize((120, 120)).convert("RGB")
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    prediction = model.predict(img_array)[0][0]
    label = "Crack Detected" if prediction > 0.5 else "No Crack"
    return label, prediction

# Crack highlighting (overlay) function
def highlight_crack_pil(img_pil):
    img_array = np.array(img_pil.convert("RGB"))
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    overlay = img_bgr.copy()
    overlay[edges > 0] = [0, 0, 255]
    img_highlighted = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_highlighted)

# Initialize history in session state
if "history" not in st.session_state:
    st.session_state["history"] = []

# App Title and Description
st.title("Concrete Crack Detection Web App 🧱")
st.markdown("""
Upload one or more images of concrete surfaces to detect cracks using a trained AI model.  
The app highlights detected cracks in red on the output image.""")

# Expandable About Section
with st.expander("About this app"):
    st.markdown("""
- Developed using TensorFlow/Keras for deep learning crack detection.  
- Crack highlighting uses OpenCV edge detection.  
- Built with Streamlit for rapid web deployment.  
- Session history tracks uploaded images and prediction results.  
    """)

# File uploader
uploaded_files = st.file_uploader(
    "Choose image files",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.header(f"File: {uploaded_file.name}")

        # Open image
        img_pil = Image.open(uploaded_file)
        st.image(img_pil, caption="Original Image", width=350)

        # Predict
        label, prob = predict_crack_pil(img_pil)

        # Display prediction with colored badge
        if label == "Crack Detected":
            st.success(f"Prediction: {label} (probability: {prob:.2f})")
        else:
            st.info(f"Prediction: {label} (probability: {prob:.2f})")

        # Add to history
        st.session_state.history.append({
            "Filename": uploaded_file.name,
            "Result": label,
            "Probability": f"{prob:.2f}"
        })

        # Highlighted crack image if detected
        if label == "Crack Detected":
            st.write("Crack area highlighted in red:")
            highlighted_img = highlight_crack_pil(img_pil)
            st.image(highlighted_img, caption="Highlighted Crack", width=350)

# Show testing history
with st.expander("🕑 View Image Analysis History"):
    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df)
    else:
        st.write("No images tested yet.")

# Footer with copyright
footer_html = """
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #f0f2f6;
    color: #666666;
    text-align: center;
    font-size: 12px;
    padding: 8px 0;
    border-top: 1px solid #e6e6e6;
    z-index: 1000;
}
</style>
<div class="footer">
    © 2025 Prince Osei Boateng. All rights reserved.
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)
