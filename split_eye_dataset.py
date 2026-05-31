"""Split eye state train folder into train/val sets.

Usage:
    python split_eye_dataset.py

Expects:
    datasets/eye_state/train/open/    ...images...
    datasets/eye_state/train/closed/  ...images...

Produces:
    datasets/eye_state/train/open/    (80%)
    datasets/eye_state/train/closed/  (80%)
    datasets/eye_state/val/open/      (20%)
    datasets/eye_state/val/closed/    (20%)
"""

import os
import shutil
import random
from pathlib import Path

DATASET_DIR  = Path("datasets/eye_state")
TRAIN_DIR    = DATASET_DIR / "train"
VAL_DIR      = DATASET_DIR / "val"
VAL_SPLIT    = 0.2
RANDOM_SEED  = 42
CLASSES      = ["open", "closed"]

random.seed(RANDOM_SEED)

for cls in CLASSES:
    src_dir = TRAIN_DIR / cls
    val_dir = VAL_DIR   / cls

    if not src_dir.exists():
        print(f"Skipping '{cls}' — folder not found: {src_dir}")
        continue

    images = [
        f for f in src_dir.iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    ]

    if not images:
        print(f"No images found in {src_dir}")
        continue

    random.shuffle(images)
    val_count  = int(len(images) * VAL_SPLIT)
    val_images = images[:val_count]

    val_dir.mkdir(parents=True, exist_ok=True)

    for img_path in val_images:
        shutil.move(str(img_path), str(val_dir / img_path.name))

    remaining = len(images) - val_count
    print(f"  {cls}: {remaining} train  |  {val_count} val")

print("\nDone. Dataset split complete.")
