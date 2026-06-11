import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model


def build_resnet50_model(num_classes):
    """
    Build ResNet50 transfer learning model for multi-class dog breed classification.
    """

    base_model = ResNet50(
        weights="imagenet",
        include_top=False, #Do not include original ImageNet final classification layer
        input_shape=(224, 224, 3)
    )

    base_model.trainable = False #Freeze base model layers to prevent training

    x = base_model.output # x contains the feature map produced by ResNet50

    x = GlobalAveragePooling2D()(x)

    x = Dense(256, activation="relu")(x) #It helps the model learn complex differences between breeds.

    x = Dropout(0.5)(x)

    output = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output)# ResNet50 base +our custom classification head

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    model = build_resnet50_model(num_classes=120)
    model.summary()