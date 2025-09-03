import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf


# Load the trained model with error handling
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model("cnn_fashion_mnist.h5")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Class names as per Fashion MNIST
class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

st.title("Fashion MNIST Classifier")
st.write("Upload one or more 28x28 grayscale images of fashion items.")

uploaded_files = st.file_uploader(
    "Choose image files", type=["png", "jpg", "jpeg"], accept_multiple_files=True
)

if uploaded_files:
    images = []
    for uploaded_file in uploaded_files:
        try:
            img = Image.open(uploaded_file).convert("L").resize((28, 28))
            arr = np.array(img, dtype="float32") / 255.0
            images.append(arr)
        except Exception as e:
            st.error(f"Error processing image {uploaded_file.name}: {e}")
    try:
        images_np = np.stack(images)[..., np.newaxis]  # (N,28,28,1)
        if model is not None:
            # Predict
            probs = model.predict(images_np)
            preds = np.argmax(probs, axis=1)
            pred_labels = [class_names[i] for i in preds]

            # Display results
            st.write("### Results")
            for i, (img, label) in enumerate(zip(images, pred_labels)):
                st.image(img, width=100, caption=f"Predicted: {label}")
        else:
            st.warning("Model is not loaded. Cannot make predictions.")
    except Exception as e:
        st.error(f"Error during prediction: {e}")