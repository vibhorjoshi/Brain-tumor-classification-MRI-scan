from flask import Flask, render_template, request
from io import BytesIO
from PIL import Image
import streamlit as st
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Function to load and preprocess an image
def preprocess_image(image):
    image = np.array(image)
    image = tf.image.resize(image, (224, 224))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = tf.expand_dims(image, axis=0)
    return image

# Load the pre-trained brain tumor classification model
model = tf.keras.models.load_model('brain_tumor_model.h5')

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        uploaded_file = request.files["file"]
        if uploaded_file.filename != "":
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Preprocess the image
            preprocessed_image = preprocess_image(image)

            # Make predictions
            predictions = model.predict(preprocessed_image)

            # Display the prediction results
            if predictions[0] > 0.5:
                result = "Tumor Detected"
            else:
                result = "No Tumor Detected"

            return render_template("result.html", result=result)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
