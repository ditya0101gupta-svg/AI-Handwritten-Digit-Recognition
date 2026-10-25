
import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="AI Handwritten Digit Recognition",
    page_icon="🔢",
    layout="centered"
)

# Load trained CNN model
@st.cache_resource
def load_cnn_model():
    model_path = Path(__file__).resolve().parent / "models" / "mnist_cnn_model.keras"
    return tf.keras.models.load_model(model_path)

model = load_cnn_model()

# Application title
st.title("🔢 AI-Based Handwritten Digit Recognition")

st.write(
    "Upload an image containing a handwritten digit (0–9). "
    "Our CNN model will recognize the digit and display its confidence score."
)

st.divider()

# Image upload
uploaded_file = st.file_uploader(
    "Upload a handwritten digit image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("L")

    st.image(
        image,
        caption="Uploaded Handwritten Digit",
        width=250
    )


    # Convert image to grayscale and improve contrast
    image = ImageOps.autocontrast(image)

    # MNIST uses white digits on black backgrounds
    if np.array(image).mean() > 127:
        image = ImageOps.invert(image)

    # Remove faint background noise
    image_array = np.array(image)
    image_array[image_array < 30] = 0
    image = Image.fromarray(image_array)

    # Find the handwritten digit boundaries
    bbox = image.getbbox()

    if bbox is not None:
        image = image.crop(bbox)

    # Resize while preserving aspect ratio
    image.thumbnail((20, 20), Image.Resampling.LANCZOS)

    # Center the digit on a 28x28 black canvas
    canvas = Image.new("L", (28, 28), color=0)

    x = (28 - image.width) // 2
    y = (28 - image.height) // 2

    canvas.paste(image, (x, y))

    # Normalize and reshape for CNN
    img_array = np.array(canvas).astype("float32") / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)


    # and center the digit on a 28x28 canvas
    image = ImageOps.autocontrast(image)

    # MNIST expects light digits on a dark background
    # Invert images with predominantly light backgrounds
    if np.array(image).mean() > 127:
        image = ImageOps.invert(image)

    # Fit image within a 20x20 region
    image.thumbnail((20, 20), Image.Resampling.LANCZOS)

    canvas = Image.new("L", (28, 28), color=0)
    x = (28 - image.width) // 2
    y = (28 - image.height) // 2
    canvas.paste(image, (x, y))

    # Normalize and reshape for CNN
    img_array = np.array(canvas).astype("float32") / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    st.image(
        canvas.resize((224, 224)),
        caption="Processed Image (CNN Input)",
        width=224
    )

    if st.button("Predict Digit", type="primary"):

        prediction = model.predict(img_array, verbose=0)

        predicted_digit = int(np.argmax(prediction))
        confidence = float(np.max(prediction)) * 100

        st.success(f"Predicted Digit: {predicted_digit}")

        st.metric(
            label="Model Confidence",
            value=f"{confidence:.2f}%"
        )

        
        st.subheader("Prediction Probabilities")

        probabilities = prediction[0] * 100

        import pandas as pd

        probability_df = pd.DataFrame({
            "Digit": [str(i) for i in range(10)],
            "Probability (%)": probabilities
        })

        st.bar_chart(
            probability_df,
            x="Digit",
            y="Probability (%)"
        )

st.divider()

st.caption(
    "Developed using Python, TensorFlow, Keras, "
    "Convolutional Neural Networks, and Streamlit."
)
