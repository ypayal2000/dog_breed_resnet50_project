import json
import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input


MODEL_PATH = "models/dog_breed_resnet50.h5"
CLASS_NAMES_PATH = "models/class_names.json"
IMAGE_SIZE = (224, 224)


st.set_page_config(
    page_title="Dog Breed Classifier",
    page_icon="🐶",
    layout="wide"
)


@st.cache_resource
def load_model():
    """
    Load trained model only once.
    This avoids reloading model again and again on every Streamlit refresh.
    """
    model = tf.keras.models.load_model(MODEL_PATH)
    return model


@st.cache_data
def load_class_names():
    """
    Load class names from JSON file.
    """
    with open(CLASS_NAMES_PATH, "r") as file:
        class_names = json.load(file)

    return class_names


def preprocess_uploaded_image(uploaded_image):
    """
    Convert uploaded image into ResNet50 input format.
    """

    image = Image.open(uploaded_image).convert("RGB")

    resized_image = image.resize(IMAGE_SIZE)

    image_array = np.array(resized_image)

    image_array = np.expand_dims(image_array, axis=0)

    image_array = preprocess_input(image_array)

    return image, image_array


def predict_top_5(model, image_array, class_names):
    """
    Predict top 5 dog breeds.
    """

    predictions = model.predict(image_array)

    top_5_indices = np.argsort(predictions[0])[-5:][::-1]

    results = []

    for index in top_5_indices:
        breed_name = class_names[index]
        confidence = float(predictions[0][index]) * 100

        results.append({
            "Breed": breed_name,
            "Confidence (%)": round(confidence, 2)
        })

    return results


def main():
    st.title("🐶 Dog Breed Classification using ResNet50")
    st.write("Upload a dog image and the model will predict the dog breed.")

    model = load_model()
    class_names = load_class_names()

    uploaded_file = st.file_uploader(
        "Upload Dog Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        original_image, image_array = preprocess_uploaded_image(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Uploaded Image")
            st.image(original_image, use_container_width=True)

        with col2:
            st.subheader("Prediction Result")

            if st.button("Predict Dog Breed"):
                with st.spinner("Predicting..."):
                    results = predict_top_5(
                        model=model,
                        image_array=image_array,
                        class_names=class_names
                    )

                top_prediction = results[0]

                st.success(
                    f"Predicted Breed: {top_prediction['Breed']}"
                )

                st.metric(
                    label="Confidence",
                    value=f"{top_prediction['Confidence (%)']}%"
                )

                st.subheader("Top 5 Predictions")
                st.dataframe(results, use_container_width=True)


if __name__ == "__main__":
    main()