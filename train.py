import os
import json
import tensorflow as tf

from src.data_loader import load_datasets
from src.model_builder import build_resnet50_model
from src.plot_utils import plot_training_history


MODEL_DIR = "models"
MODEL_PATH = "models\\dog_breed_resnet50.h5"
CLASS_NAMES_PATH = "models\\class_names.json"

EPOCHS = 5


def train_model():
    """
    Train ResNet50 transfer learning model for dog breed classification.
    """

    print("\n==============================")
    print("Loading datasets...")
    print("==============================")

    train_dataset, test_dataset, class_names = load_datasets()

    num_classes = len(class_names)

    print("\nTotal dog breed classes:", num_classes)

    print("\n==============================")
    print("Building ResNet50 model...")
    print("==============================")

    model = build_resnet50_model(num_classes=num_classes)

    model.summary()

    print("\n==============================")
    print("Starting model training...")
    print("==============================")

    history = model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=EPOCHS
    )

    print("\n==============================")
    print("Saving model...")
    print("==============================")

    os.makedirs(MODEL_DIR, exist_ok=True)

    model.save(MODEL_PATH)

    with open(CLASS_NAMES_PATH, "w") as file:
        json.dump(class_names, file)

    print("Model saved at:", MODEL_PATH)
    print("Class names saved at:", CLASS_NAMES_PATH)

    print("\n==============================")
    print("Plotting training graphs...")
    print("==============================")

    accuracy_plot_path, loss_plot_path = plot_training_history(history)

    print("Accuracy plot saved at:", accuracy_plot_path)
    print("Loss plot saved at:", loss_plot_path)

    print("\n==============================")
    print("Training completed successfully")
    print("==============================")


if __name__ == "__main__":
    train_model()