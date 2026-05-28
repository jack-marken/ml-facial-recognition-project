"""Evaluate the classification-based face recognition model using ROC/AUC (Karam).

Reads verification_pairs_val.txt and computes cosine similarity scores for
each trial, then plots the ROC curve and prints the AUC score.

Usage:
    python -m face_verification.classification_model.evaluate_karam \\
        --pairs     datasets/verification_pairs_val.txt \\
        --model     models/recognition_classification_karam.pth \\
        --gallery   models/recognition_gallery_classification_karam.pkl
"""
# Author: Karam

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve

from .classification_model_karam import generate_embedding, load_classification_model
from .recognition_karam import _cosine_similarity, _get_gallery


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ROC/AUC evaluation for classification face recognition (Karam)."
    )
    parser.add_argument("--pairs",   default="datasets/verification_pairs_val.txt")
    parser.add_argument("--model",   default="models/recognition_classification_karam.pth")
    parser.add_argument("--gallery", default="models/recognition_gallery_classification_karam.pkl")
    parser.add_argument("--plot",    default="models/roc_classification_karam.png",
                        help="Path to save the ROC curve image.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model   = load_classification_model(model_path=args.model)
    gallery = _get_gallery(Path(args.gallery))

    pairs_path = Path(args.pairs)
    if not pairs_path.exists():
        raise FileNotFoundError(f"Pairs file not found: {pairs_path}")

    scores: list[float] = []
    labels: list[int]   = []
    skipped = 0

    with pairs_path.open() as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) != 3:
                continue

            path_a, path_b, label = parts[0], parts[1], int(parts[2])

            embedding_a = _load_embedding(path_a, model)
            embedding_b = _load_embedding(path_b, model)

            if embedding_a is None or embedding_b is None:
                skipped += 1
                continue

            scores.append(_cosine_similarity(embedding_a, embedding_b))
            labels.append(label)

    if not scores:
        print("No valid pairs evaluated — check image paths in pairs file.")
        return

    scores_arr = np.array(scores, dtype="float32")
    labels_arr = np.array(labels, dtype="int32")

    fpr, tpr, _ = roc_curve(labels_arr, scores_arr)
    roc_auc     = auc(fpr, tpr)

    print(f"\n=== Classification Model Evaluation (Karam) ===")
    print(f"Pairs evaluated : {len(scores)}")
    print(f"Pairs skipped   : {skipped}")
    print(f"AUC             : {roc_auc:.4f}")

    # Save ROC curve plot
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color="steelblue", lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Classification Face Recognition (Karam)")
    plt.legend(loc="lower right")
    plt.tight_layout()

    plot_path = Path(args.plot)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=150)
    print(f"ROC curve saved → {plot_path}")
    plt.close()


def _load_embedding(image_path: str, model) -> np.ndarray | None:
    frame = cv2.imread(image_path)
    if frame is None:
        return None

    resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return generate_embedding(model, rgb)


if __name__ == "__main__":
    main()
