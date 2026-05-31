"""Train fatigue/drowsiness detection model (Karam — D/HD Innovative Feature).

Dataset: Yawn Eye Dataset
  https://www.kaggle.com/datasets/serenaraju/yawn-eye-dataset-new

4 classes covering all fatigue indicators:
  Closed   → eyes shut     → DROWSY signal
  Open     → eyes open     → ALERT
  Yawn     → yawning       → DROWSY signal
  no_yawn  → normal face   → ALERT

Expected dataset layout:
    datasets/yawn_eye/
        train/
            Closed/    Open/    Yawn/    no_yawn/
        test/
            Closed/    Open/    Yawn/    no_yawn/

Usage:
    python -m fatigue_detection.train_fatigue_karam
"""
# Author: Karam (Innovative Feature — D/HD)

from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator


DATASET_DIR = Path("datasets/yawn_eye")
DEFAULT_OUT = "models/fatigue_karam.h5"
IMAGE_SIZE  = (224, 224)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train fatigue detection model on Yawn Eye Dataset (Karam)."
    )
    parser.add_argument("--data-dir",      default=str(DATASET_DIR))
    parser.add_argument("--output",        default=DEFAULT_OUT)
    parser.add_argument("--epochs",        type=int,   default=20)
    parser.add_argument("--batch-size",    type=int,   default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    return parser.parse_args()


def main() -> None:
    args      = parse_args()
    data_dir  = Path(args.data_dir)
    train_dir = data_dir / "train"
    test_dir  = data_dir / "test"

    if not train_dir.exists():
        print(f"\nERROR: Could not find {train_dir}")
        print("Make sure the dataset is placed at:")
        print("  datasets/yawn_eye/train/Closed/")
        print("  datasets/yawn_eye/train/Open/")
        print("  datasets/yawn_eye/train/Yawn/")
        print("  datasets/yawn_eye/train/no_yawn/\n")
        return

    # ── Data generators ───────────────────────────────────────────────────
    train_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        zoom_range=0.1,
        validation_split=0.1,
    )
    val_gen  = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.1)
    test_gen = ImageDataGenerator(rescale=1.0 / 255)

    train_data = train_gen.flow_from_directory(
        train_dir, target_size=IMAGE_SIZE, batch_size=args.batch_size,
        class_mode="categorical", subset="training", shuffle=True,
    )
    val_data = val_gen.flow_from_directory(
        train_dir, target_size=IMAGE_SIZE, batch_size=args.batch_size,
        class_mode="categorical", subset="validation", shuffle=False,
    )
    test_data = test_gen.flow_from_directory(
        test_dir, target_size=IMAGE_SIZE, batch_size=args.batch_size,
        class_mode="categorical", shuffle=False,
    )

    print(f"\nClasses : {train_data.class_indices}")
    print(f"Train   : {train_data.samples}")
    print(f"Val     : {val_data.samples}")
    print(f"Test    : {test_data.samples}\n")

    # ── Model ─────────────────────────────────────────────────────────────
    base = MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    for layer in base.layers[:-20]:
        layer.trainable = False

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(len(train_data.class_indices), activation="softmax")(x)

    model = Model(inputs=base.input, outputs=x)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary(line_length=80)

    # ── Train ─────────────────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            str(output_path), monitor="val_accuracy",
            save_best_only=True, verbose=1,
        ),
        EarlyStopping(
            monitor="val_accuracy", patience=5,
            restore_best_weights=True, verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=3, verbose=1,
        ),
    ]

    model.fit(
        train_data,
        validation_data=val_data,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    # ── Evaluate ──────────────────────────────────────────────────────────
    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(test_data, verbose=1)
    print(f"\nTest accuracy : {test_acc:.4f}")

    # Save class index map used by fatigue_karam.py at inference
    class_map_path = output_path.parent / "fatigue_class_indices_karam.json"
    with class_map_path.open("w") as f:
        json.dump(train_data.class_indices, f, indent=2)

    print(f"Model saved      → {output_path}")
    print(f"Class map saved  → {class_map_path}")
    print("\nDone! Run:  python test_fatigue_karam.py")


if __name__ == "__main__":
    main()
