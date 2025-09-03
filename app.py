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
st.write("Upload one or more images (PNG, JPG, JPEG, WEBP) of fashion items.\nAny size or color image is accepted and will be automatically converted to 28x28 grayscale for prediction.")


# Option 1: User uploads their own images
uploaded_files = st.file_uploader(
    "Choose image files", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True
)

# Option 2: User selects from sample images if no upload
import os
sample_dir = "fashion_pieces_10"
sample_files = [f for f in os.listdir(sample_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
sample_paths = [os.path.join(sample_dir, f) for f in sample_files]

selected_sample_files = []
if not uploaded_files:
    st.write("Or select from sample images:")
    selected = st.multiselect(
        "Sample images", options=sample_files, default=[]
    )
    selected_sample_files = [os.path.join(sample_dir, f) for f in selected]

user_files = uploaded_files if uploaded_files else selected_sample_files
user_filenames = [f.name if hasattr(f, 'name') else os.path.basename(f) for f in user_files]

# Prepare images from either uploads or selected samples
processed_images = []  # normalized arrays for model input
display_images = []    # resized grayscale PIL images for display

# Choose resampling filter compatible with Pillow version
try:
    RESAMPLE = Image.Resampling.LANCZOS  # PIL >= 10
except AttributeError:
    RESAMPLE = Image.LANCZOS            # PIL < 10

for user_file in user_files:
    try:
        img = Image.open(user_file)
        img_gray = img.convert("L")
        img_resized = img_gray.resize((28, 28), RESAMPLE)
        arr = np.array(img_resized, dtype="float32") / 255.0
        processed_images.append(arr)
        display_images.append(img_resized)
    except Exception as e:
        name = getattr(user_file, 'name', str(user_file))
        st.error(f"Error processing image {name}: {e}")


# Display selected images in a responsive grid
st.write("### Selected Images (Resized to 28x28 Grayscale)")
if display_images:
    n_images = len(display_images)
    n_cols = min(5, n_images) if n_images > 1 else 1
    cols = st.columns(n_cols)
    for idx, (img_disp, fname) in enumerate(zip(display_images, user_filenames)):
        with cols[idx % n_cols]:
            st.image(img_disp, width=100, caption=f"{fname}")
else:
    st.info("No images selected yet. Upload files above or pick from sample images.")

# Classification code
if st.button("Classify Images"):
    if not processed_images:
        st.warning("No images to classify. Please upload or select sample images.")
    elif model is None:
        st.warning("Model is not loaded. Cannot make predictions.")
    else:
        try:
            images_np = np.stack(processed_images)[..., np.newaxis]  # (N,28,28,1)
            probs = model.predict(images_np)
            preds = np.argmax(probs, axis=1)
            pred_labels = [class_names[i] for i in preds]

            st.write("### Results")
            n_images = len(display_images)
            n_cols = min(5, n_images) if n_images > 1 else 1
            cols = st.columns(n_cols)
            for idx, (img_disp, label, fname) in enumerate(zip(display_images, pred_labels, user_filenames)):
                with cols[idx % n_cols]:
                    st.image(img_disp, width=100, caption=f"{fname} | Predicted: {label}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")