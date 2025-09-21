import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

# --- Page Configuration ---
st.set_page_config(
    page_title="SAR Oil Spill Detection",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Model Loading ---
@st.cache(allow_output_mutation=True)
def load_model(model_path: str):
    """Loads the pre-trained TensorFlow model."""
    try:
        model = tf.keras.models.load_model(model_path)
        st.success("U-Net model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Provide the path to your best-trained model
MODEL_PATH = "models/unet_model.h5" # Make sure to save your model here
model = load_model(MODEL_PATH)


# --- UI and Logic ---
def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocesses the uploaded image to match model input requirements."""
    img = np.array(image.convert('L')) # Convert to grayscale
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LANCZOS4)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=-1) # Add channel dimension
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def display_prediction(original_image, prediction_mask):
    """Displays the original image and the predicted mask side-by-side."""
    mask = np.squeeze(prediction_mask) * 255.0
    mask = mask.astype(np.uint8)

    # Create an overlay
    overlay = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
    colored_mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    overlayed_image = cv2.addWeighted(colored_mask, 0.5, overlay, 0.5, 0)


    col1, col2 = st.columns(2)
    with col1:
        st.image(original_image, caption="Original SAR Image", use_column_width=True)
    with col2:
        st.image(overlayed_image, caption="Predicted Oil Spill Mask", use_column_width=True, channels="BGR")


# --- Main App ---
st.title("🛰️ SAR Image Oil Spill Segmentation")
st.markdown(
    """
    Upload a SAR image to detect and segment potential oil spills using a pre-trained U-Net model.
    """
)

uploaded_file = st.file_uploader(
    "Choose a SAR image...", type=["jpg", "jpeg", "png", "tif"]
)

if uploaded_file is not None:
    # Open the image
    pil_image = Image.open(uploaded_file)
    
    st.info("Image uploaded successfully. Processing...")

    if model is not None:
        # Preprocess and predict
        processed_image = preprocess_image(pil_image)
        prediction = model.predict(processed_image)

        # Display results
        st.subheader("Segmentation Results")
        display_prediction(pil_image.resize((512, 512)), prediction)
    else:
        st.warning("Model not loaded. Cannot perform prediction.")

# --- Sidebar ---
st.sidebar.header("About")
st.sidebar.info(
    "This web application uses a U-Net deep learning model to perform "
    "semantic segmentation on SAR images to identify oil spills."
)
st.sidebar.header("How to Use")
st.sidebar.markdown(
    "1. **Upload an image**: Use the uploader in the main panel.\n"
    "2. **View the results**: The model will automatically process the image and display the segmentation mask."
)
