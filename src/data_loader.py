import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input


IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

TRAIN_DIR = "data\\train"
TEST_DIR = "data\\test"


def load_datasets():
    """
    Load train and test datasets from folder structure.
    """

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=True
    )

    test_dataset = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False
    )

    class_names = train_dataset.class_names

    train_dataset = train_dataset.map(
        lambda images, labels: (preprocess_input(images), labels)
    )

    test_dataset = test_dataset.map(
        lambda images, labels: (preprocess_input(images), labels)
    )

    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

    return train_dataset, test_dataset, class_names


if __name__ == "__main__":
    train_dataset, test_dataset, class_names = load_datasets()

    print("Datasets loaded successfully")
    print("Total classes:", len(class_names))
    print("First 10 classes:", class_names[:10])

    for images, labels in train_dataset.take(1):
        print("Image batch shape:", images.shape)
        print("Label batch shape:", labels.shape)