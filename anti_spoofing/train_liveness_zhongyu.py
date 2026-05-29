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
    from .reporting_liveness_zhongyu import save_liveness_report
except ImportError:
    from model_factory_zhongyu import build_liveness_model
    from reporting_liveness_zhongyu import save_liveness_report


def parse_args(
    default_architecture: str | None = None,
    default_output: str | None = None,
    default_save_weights_only: bool = False,
) -> argparse.Namespace:
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
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-base", action="store_true")
    parser.add_argument(
        "--disable-augmentation",
        action="store_true",
        help="Disable Kaixiang-style training augmentation.",
    )
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument(
        "--output",
        default=default_output,
        help=(
            "Output .keras model path. Defaults to "
            "models/liveness_<architecture>_zhongyu.keras."
        ),
    )
    parser.add_argument(
        "--save-weights-only",
        action="store_true",
        default=default_save_weights_only,
        help="Save only model weights. Output path must end with .weights.h5.",
    )
    parser.add_argument(
        "--report-dir",
        default="anti_spoofing/report_liveness_zhongyu",
        help="Directory for training tables and figures.",
    )
    return parser.parse_args()


def main(
    default_architecture: str | None = None,
    default_output: str | None = None,
    default_save_weights_only: bool = False,
) -> None:
    args = parse_args(
        default_architecture=default_architecture,
        default_output=default_output,
        default_save_weights_only=default_save_weights_only,
    )

    import tensorflow as tf

    data_dir = Path(args.data_dir)
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"

    if not train_dir.exists() or not val_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(
            "Expected dataset folders at datasets/liveness/train, "
            "datasets/liveness/val, and datasets/liveness/test."
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
    test_dataset = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="binary",
        class_names=["spoof", "real"],
        image_size=(224, 224),
        batch_size=args.batch_size,
        shuffle=False,
    )

    autotune = tf.data.AUTOTUNE
    if not args.disable_augmentation:
        augmentation = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomZoom(height_factor=(-0.12, 0.0), width_factor=(-0.12, 0.0)),
                tf.keras.layers.RandomRotation(0.045),
                tf.keras.layers.RandomContrast(0.2),
                tf.keras.layers.RandomBrightness(0.2),
            ],
            name="kaixiang_style_liveness_augmentation",
        )
        train_dataset = train_dataset.map(
            lambda images, labels: (augmentation(images, training=True), labels),
            num_parallel_calls=autotune,
        )

    train_dataset = train_dataset.prefetch(autotune)
    val_dataset = val_dataset.prefetch(autotune)
    test_dataset = test_dataset.prefetch(autotune)

    model = build_liveness_model(
        architecture=args.architecture,
        train_base=args.train_base,
    )
    try:
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    except AttributeError:
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    output_path = Path(args.output or f"models/liveness_{args.architecture}_zhongyu.keras")
    if args.save_weights_only and not output_path.name.endswith(".weights.h5"):
        raise ValueError("When --save-weights-only is used, output must end with .weights.h5.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=output_path,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=args.save_weights_only,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.early_stopping_patience,
            restore_best_weights=True,
        ),
    ]

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
        callbacks=callbacks,
    )
    if args.save_weights_only:
        model.save_weights(output_path)
    else:
        model.save(output_path)
    print(f"Saved liveness model to {output_path}")
    save_liveness_report(
        model=model,
        history=history,
        args=args,
        output_path=output_path,
        data_dir=data_dir,
        report_root=Path(args.report_dir),
        test_dataset=test_dataset,
    )
    print(f"Saved liveness report artifacts to {Path(args.report_dir) / args.architecture}")


if __name__ == "__main__":
    main()
