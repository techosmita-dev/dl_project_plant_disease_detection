import os
import tensorflow as tf
import json
from PIL import Image
import numpy as np
import streamlit as st

# Get the current working directory (the directory containing this file)
working_dir = os.path.dirname(os.path.abspath(__file__))

# Define model path (relative to this file)
model_path = os.path.join(working_dir, "trained_model", "trained_disease_model.h5")

# Try loading the model safely
model = None
if not os.path.exists(model_path):
    st.error(f"❌ Model file not found at: {model_path}")
else:
    try:
        model = tf.keras.models.load_model(model_path)
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"⚠️ Failed to load model: {e}")

# Load class indices (if available)
class_indices_path = os.path.join(working_dir, "class_indices.json")
if not os.path.exists(class_indices_path):
    st.error("❌ class_indices.json not found!")
else:
    class_indices = json.load(open(class_indices_path))


# Function to Load and Preprocess the Image
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).resize(target_size)
    img_array = np.expand_dims(np.array(img).astype('float32') / 255., axis=0)
    return img_array


# Prediction Function
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name


# Streamlit App UI
st.title('🌿 Plant Disease Classifier')

uploaded_image = st.file_uploader("📸 Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        st.image(image.resize((150, 150)), caption="Uploaded Image")

    with col2:
        if model is not None and st.button('🔍 Classify'):
            prediction = predict_image_class(model, uploaded_image, class_indices)
            st.success(f'✅ Prediction: **{prediction}**')
        elif model is None:
            st.warning("⚠️ Model not loaded — check your model path or file.")
