import json
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input


MODEL_PATH = "models\\dog_breed_resnet50.h5"
CLASS_NAMES_PATH = "models\\class_names.json"
IMAGE_SIZE = (224, 224)


def load_class_names():
    """
    Load dog breed class names.
    """

    with open(CLASS_NAMES_PATH, "r") as file:
        class_names = json.load(file)

    return class_names


def load_and_preprocess_image(image_path):
    """
    Load image, resize it to 224x224, convert it into array,
    add batch dimension, and apply ResNet50 preprocessing.
    """

    img = image.load_img(image_path, target_size=IMAGE_SIZE)

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)

    img_array = preprocess_input(img_array)

    return img_array


def predict_dog_breed(image_path):
    """
    Predict dog breed from input image.
    """

    print("\n==============================")
    print("Loading trained model...")
    print("==============================")

    model = tf.keras.models.load_model(MODEL_PATH)

    class_names = load_class_names()

    print("Model loaded successfully")
    print("Total classes:", len(class_names))

    print("\n==============================")
    print("Preprocessing image...")
    print("==============================")

    img_array = load_and_preprocess_image(image_path)

    print("Image shape:", img_array.shape)

    print("\n==============================")
    print("Predicting dog breed...")
    print("==============================")

    predictions = model.predict(img_array)

    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    confidence = predictions[0][predicted_index]

    print("\nFinal Prediction")
    print("------------------------------")
    print("Predicted Breed:", predicted_class)
    print("Confidence:", round(float(confidence) * 100, 2), "%")

    print("\nTop 5 Predictions")
    print("------------------------------")

    top_5_indices = np.argsort(predictions[0])[-5:][::-1]

    for index in top_5_indices:
        breed_name = class_names[index]
        breed_confidence = predictions[0][index]

        print(f"{breed_name}: {round(float(breed_confidence) * 100, 2)}%")


if __name__ == "__main__":
    image_path = "test_images\\dog_h.jpg"
    predict_dog_breed(image_path)