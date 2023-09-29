# Import necessary libraries
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Function to load and preprocess an image
def preprocess_image(image):
    image = np.array(image)
    image = tf.image.resize(image, (224, 224))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = tf.expand_dims(image, axis=0)
    return image

# Load the pre-trained brain tumor classification model
model = tf.keras.models.load_model('keras_model.h5')

# Set the Streamlit app title
st.title('Brain Tumor Classification')

# Upload an image for classification
uploaded_image = st.file_uploader("Upload an MRI image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Make predictions
    predictions = model.predict(preprocessed_image)

    # Display the prediction results
    if predictions[0] > 0.5:
        st.write("Prediction: Tumor Detected")
    else:
        st.write("Prediction: No Tumor Detected")
