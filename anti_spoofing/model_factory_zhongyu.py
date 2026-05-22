"""Model builders for liveness binary classification."""

from __future__ import annotations


def build_liveness_model(
    architecture: str,
    input_shape: tuple[int, int, int] = (224, 224, 3),
    train_base: bool = False,
    dropout_rate: float = 0.3,
):
    """Build a Keras transfer-learning model for real/spoof classification."""
    from tensorflow.keras import layers, models

    normalized_architecture = architecture.lower()

    if normalized_architecture == "resnet50v2":
        from tensorflow.keras.applications import ResNet50V2

        base_model = ResNet50V2(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
        )
    elif normalized_architecture == "densenet121":
        from tensorflow.keras.applications import DenseNet121

        base_model = DenseNet121(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
        )
    else:
        raise ValueError(
            "Unsupported architecture. Use 'resnet50v2' or 'densenet121'."
        )

    base_model.trainable = train_base

    inputs = layers.Input(shape=input_shape)
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="scale_rgb_to_minus1_1")(
        inputs
    )
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="real_probability")(x)

    return models.Model(inputs=inputs, outputs=outputs, name=f"liveness_{architecture}")
