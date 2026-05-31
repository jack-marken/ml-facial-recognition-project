"""Train fatigue/drowsiness detection model — auto-downloads dataset (Karam D/HD).

Automatically downloads the Yawn Eye Dataset from Kaggle.
No manual downloading required.

You only need your Kaggle credentials (asked once, inline):
  1. Go to https://www.kaggle.com/settings
  2. Scroll to API section → click "Create New Token"
  3. A kaggle.json file downloads — open it, copy username and key
  4. Paste them when this script asks

4 fatigue indicators detected:
  Closed   → eyes shut        → DROWSY signal
  Open     → eyes open        → ALERT
  Yawn     → mouth yawning    → DROWSY signal
  no_yawn  → normal face      → ALERT

Usage:
    python -m fatigue_detection.train_fatigue_karam
"""
# Author: Karam (Innovative Feature — D/HD)

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

DATASET_URL   = "https://www.kaggle.com/datasets/serenaraju/yawn-eye-dataset-new"
DATASET_DIR   = Path("datasets/yawn_eye")
DOWNLOAD_DIR  = Path("datasets")
DEFAULT_OUT   = "models/fatigue_karam.h5"
IMAGE_SIZE    = (224, 224)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train fatigue detection model — auto-downloads dataset (Karam)."
    )
    parser.add_argument("--output",        default=DEFAULT_OUT)
    parser.add_argument("--epochs",        type=int,   default=20)
    parser.add_argument("--batch-size",    type=int,   default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── Step 1: Auto-download dataset ────────────────────────────────────
    _ensure_dataset()

    # ── Step 2: Build data generators ────────────────────────────────────
    import tensorflow as tf
    from tensorflow.keras import layers, Model
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_dir = DATASET_DIR / "train"
    test_dir  = DATASET_DIR / "test"

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
    print(f"Train   : {train_data.samples}  Val: {val_data.samples}  Test: {test_data.samples}\n")

    # ── Step 3: Build MobileNetV2 model ───────────────────────────────────
    base = MobileNetV2(input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet")
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

    # ── Step 4: Train ─────────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(str(output_path), monitor="val_accuracy",
                        save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_accuracy", patience=5,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
    ]

    model.fit(train_data, validation_data=val_data,
              epochs=args.epochs, callbacks=callbacks)

    # ── Step 5: Evaluate ──────────────────────────────────────────────────
    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(test_data, verbose=1)
    print(f"\nTest accuracy: {test_acc:.4f}")

    # Save class index map for inference
    class_map_path = output_path.parent / "fatigue_class_indices_karam.json"
    with class_map_path.open("w") as f:
        json.dump(train_data.class_indices, f, indent=2)

    print(f"\nModel saved      → {output_path}")
    print(f"Class map saved  → {class_map_path}")
    print("\nDone! Now run:  python test_fatigue_karam.py")


def _ensure_dataset() -> None:
    """Download and organise the Yawn Eye Dataset if not already present."""

    train_dir = DATASET_DIR / "train"
    if train_dir.exists() and any(train_dir.iterdir()):
        print(f"Dataset already exists at {DATASET_DIR} — skipping download.\n")
        return

    print("=" * 60)
    print("Yawn Eye Dataset not found — downloading from Kaggle.")
    print("=" * 60)
    print("\nYou need your Kaggle API credentials:")
    print("  1. Go to https://www.kaggle.com/settings")
    print("  2. Scroll to 'API' section → click 'Create New Token'")
    print("  3. Open the downloaded kaggle.json and copy username + key\n")

    # Install opendatasets if needed
    try:
        import opendatasets as od
    except ImportError:
        print("Installing opendatasets...")
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "opendatasets", "-q"])
        import opendatasets as od

    # Download — asks for username + key inline
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    od.download(DATASET_URL, data_dir=str(DOWNLOAD_DIR))

    # opendatasets extracts into datasets/yawn-eye-dataset-new/
    # Rename to our expected path: datasets/yawn_eye/
    raw_folder = DOWNLOAD_DIR / "yawn-eye-dataset-new"
    if raw_folder.exists() and not DATASET_DIR.exists():
        raw_folder.rename(DATASET_DIR)
        print(f"\nDataset organised → {DATASET_DIR}")

    # Verify the expected structure
    _verify_structure()


def _verify_structure() -> None:
    """Check folder structure and fix common naming issues."""
    expected_classes = {"Closed", "Open", "Yawn", "no_yawn"}
    train_dir = DATASET_DIR / "train"
    test_dir  = DATASET_DIR / "test"

    for split_dir in [train_dir, test_dir]:
        if not split_dir.exists():
            raise RuntimeError(
                f"Expected folder not found: {split_dir}\n"
                f"Check that the dataset extracted correctly into {DATASET_DIR}"
            )
        found = {p.name for p in split_dir.iterdir() if p.is_dir()}
        missing = expected_classes - found
        if missing:
            print(f"Warning: missing class folders in {split_dir.name}/: {missing}")
        else:
            print(f"  {split_dir.name}/ — classes OK: {sorted(found)}")

    print()


if __name__ == "__main__":
    main()