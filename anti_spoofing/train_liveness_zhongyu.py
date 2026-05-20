"""Train ResNet50V2 or DenseNet121 liveness detection models.

Expected dataset layout:

datasets/liveness/
+-- train/
|   +-- spoof/
|   +-- real/
+-- val/
|   +-- spoof/
|   +-- real/
+-- test/
    +-- spoof/
    +-- real/
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .model_factory_zhongyu import build_liveness_model
except ImportError:
    from model_factory_zhongyu import build_liveness_model


def parse_args(default_architecture: str | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a liveness model.")
    parser.add_argument(
        "--architecture",
        choices=["resnet50v2", "densenet121"],
        default=default_architecture,
        required=default_architecture is None,
        help="Backbone architecture to train.",
    )
    parser.add_argument(
        "--data-dir",
        default="datasets/liveness",
        help="Dataset root containing train/val/test folders.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--train-base", action="store_true")
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output .keras model path. Defaults to "
            "models/liveness_<architecture>_zhongyu.keras."
        ),
    )
    return parser.parse_args()


def main(default_architecture: str | None = None) -> None:
    args = parse_args(default_architecture=default_architecture)

    import tensorflow as tf

    data_dir = Path(args.data_dir)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            "Expected dataset folders at datasets/liveness/train and "
            "datasets/liveness/val."
        )

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="binary",
        class_names=["spoof", "real"],
        image_size=(224, 224),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_dataset = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode="binary",
        class_names=["spoof", "real"],
        image_size=(224, 224),
        batch_size=args.batch_size,
        shuffle=False,
    )

    autotune = tf.data.AUTOTUNE
    train_dataset = train_dataset.prefetch(autotune)
    val_dataset = val_dataset.prefetch(autotune)

    model = build_liveness_model(
        architecture=args.architecture,
        train_base=args.train_base,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    output_path = Path(
        args.output or f"models/liveness_{args.architecture}_zhongyu.keras"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=output_path,
            monitor="val_accuracy",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),
    ]

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
        callbacks=callbacks,
    )
    model.save(output_path)
    print(f"Saved liveness model to {output_path}")


if __name__ == "__main__":
    main()
