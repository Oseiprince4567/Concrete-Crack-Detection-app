# Concrete-Crack-Detection-app
This app demonstrates a complete machine learning deployment pipeline for concrete crack detection, beginning with a rigorously prepared and balanced image dataset (40,000 total images, 227×227 RGB, no augmentation), moving through model training in TensorFlow/Keras, and culminating in a Streamlit-powered browser app.
Concrete Crack Detection Web App – Technical Overview

Purpose:
This web app allows users to upload one or more images of concrete surfaces and receive immediate, AI-driven crack detection and visualization through a user-friendly web interface.

Dataset Details:
- Total Dataset Size: 40,000 images.
- Negative (No Crack) Set: 20,000 images—concrete with no cracks.
- Positive (Crack) Set: 20,000 images—concrete with visible cracks.
- Each image is 227×227 pixels, RGB.

No Augmentation:
No data augmentation (such as random rotation or flipping) was applied; all images remain in their original form.

Class Structure:
Balanced classes, with equal representation of cracked and non-cracked surfaces.

Source:
Images are arranged in separate folders for each class to facilitate supervised learning.

Model Development & Preprocessing:
- Framework: TensorFlow/Keras
- The deep learning model was built and trained using the TensorFlow library with the Keras API.
- Input Preprocessing:
  Before being passed to the model, each user-uploaded image is resized to 120×120 pixels and converted to RGB.
  This standardization ensures compatibility with the model architecture and efficient processing.
- Model Script:
  Initial training, architecture design, and saving were managed within python (model available upon request).

- Prediction Logic:
  Loading and inference code, including preprocessing and optional visualization, originated in possib.py and has since been merged into the web app code for seamless use.

Web Application Layer:
- Framework: Streamlit
- The app’s UI uses Streamlit for rapid and intuitive deployment.

Features:
- Upload Interface: Allows drag-and-drop upload of one or more JPG/JPEG/PNG images.
- Preprocessing: Each uploaded image is automatically resized to 120×120 pixels before prediction.
- Inference: The crack detection model instantly predicts presence or absence of cracks in each image.
- Visualization: When cracks are found, OpenCV highlights the cracked regions in red.
- Batch Processing: All selected images are processed, predicted, and displayed in a single session.

Resource Management:
Model loaded once with Streamlit’s @st.cache_resource decorator for efficiency.

Technical Stack:

| Component       | Technology/Method              | Description                                                |
|-----------------|-------------------------------|------------------------------------------------------------|
| Dataset         | Custom concrete dataset        | 20,000 positive + 20,000 negative, 227×227 RGB, no augmentation |
| Preprocessing   | Pillow/NumPy/Streamlit         | Resizing every image to 120×120 RGB for model input         |
| Model Training  | TensorFlow/Keras              | Deep CNN/binary classifier, input 120×120×3, saved as concrete_model.h5   |
| Inference & UI  | TensorFlow, Streamlit         | Loads model, preprocesses input, displays results in browser|
| Overlay         | OpenCV                        | Edge detection, overlays crack regions in red               |

Summary:
This app demonstrates a complete machine learning deployment pipeline for concrete crack detection, beginning with a rigorously prepared and balanced image dataset (40,000 total images, 227×227 RGB, no augmentation), moving through model training in TensorFlow/Keras, and culminating in a Streamlit-powered browser app that standardizes inputs by resizing them to 120×120 pixels for robust, user-friendly inference and visualization, with no server-side image retention.
