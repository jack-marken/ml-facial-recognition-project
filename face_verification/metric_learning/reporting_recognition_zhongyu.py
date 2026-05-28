"""Report artifact helpers for Zhongyu's metric-learning recognition runs."""

from __future__ import annotations

import csv
import itertools
import json
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from .embedding_model_zhongyu import generate_embedding


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def save_recognition_report(
    *,
    model,
    history: list[dict[str, float]],
    args,
    architecture: str,
    output_path: Path,
    report_root: Path,
    embedding_size: int,
) -> None:
    """Save triplet training history and verification metrics for a model."""
    report_dir = report_root / architecture
    report_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "backbone_model": architecture,
        "embedding_size": embedding_size,
        "input_image_size": [224, 224, 3],
        "optimizer": "Adam",
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "loss_function": "TripletMarginLoss",
        "margin": args.margin,
        "distance_metric": "cosine_similarity",
        "dataset_name": str(Path(args.data_dir).parent),
        "train_sample_count": _count_images(Path(args.data_dir)),
        "validation_sample_count": _count_images(Path(args.val_dir)),
        "test_sample_count": _count_images(Path(args.test_dir)),
        "train_backbone": bool(args.train_backbone),
        "model_output": str(output_path),
    }
    _write_json(report_dir / "config.json", config)
    _write_json(report_dir / "training_history.json", history)
    _write_history_csv(report_dir / "training_history.csv", history)
    _plot_loss_curve(report_dir / "loss_curve.png", history, architecture)

    verification = evaluate_verification_pairs(
        model=model,
        test_dir=Path(args.test_dir),
        max_pairs=args.max_eval_pairs,
        seed=args.seed,
    )
    _write_json(report_dir / "verification_metrics.json", verification)
    _write_metrics_csv(report_dir / "verification_metrics.csv", verification)
    _write_results_table(report_dir / "results_table.csv", architecture, verification)
    _plot_roc_curve(
        report_dir / "roc_curve.png",
        verification["roc_curve"],
        verification["auc"],
        title=f"{architecture} Verification ROC",
    )


def evaluate_verification_pairs(
    *,
    model,
    test_dir: Path,
    max_pairs: int,
    seed: int,
) -> dict[str, Any]:
    """Evaluate same/different identity verification on labelled test folders."""
    image_items = _load_labelled_images(test_dir)
    embeddings = []
    for identity, image_path in image_items:
        face_image = _load_resized_face(image_path)
        embeddings.append((identity, image_path.name, generate_embedding(model, face_image)))

    pairs = list(itertools.combinations(range(len(embeddings)), 2))
    if max_pairs > 0 and len(pairs) > max_pairs:
        rng = random.Random(seed)
        pairs = rng.sample(pairs, max_pairs)

    labels = []
    scores = []
    for first_index, second_index in pairs:
        first_identity, _, first_embedding = embeddings[first_index]
        second_identity, _, second_embedding = embeddings[second_index]
        labels.append(1 if first_identity == second_identity else 0)
        scores.append(_cosine_similarity(first_embedding, second_embedding))

    y_true = np.asarray(labels, dtype=int)
    y_scores = np.asarray(scores, dtype=float)
    roc_curve = _roc_curve(y_true, y_scores)
    best_threshold, best_accuracy = _best_threshold(y_true, y_scores)
    predictions = (y_scores >= best_threshold).astype(int)
    positives = max(int(np.sum(y_true == 1)), 1)
    negatives = max(int(np.sum(y_true == 0)), 1)
    tp = int(np.sum((y_true == 1) & (predictions == 1)))
    fp = int(np.sum((y_true == 0) & (predictions == 1)))

    return {
        "pair_count": int(len(y_true)),
        "positive_pair_count": int(np.sum(y_true == 1)),
        "negative_pair_count": int(np.sum(y_true == 0)),
        "verification_accuracy": round(float(best_accuracy), 6),
        "best_threshold": round(float(best_threshold), 6),
        "auc": round(float(_auc(roc_curve)), 6),
        "tpr_at_best_threshold": round(float(tp / positives), 6),
        "fpr_at_best_threshold": round(float(fp / negatives), 6),
        "roc_curve": roc_curve,
    }


def _load_labelled_images(test_dir: Path) -> list[tuple[str, Path]]:
    if not test_dir.exists():
        raise FileNotFoundError(f"Recognition test directory not found: {test_dir}")

    items = []
    for identity_dir in sorted(path for path in test_dir.iterdir() if path.is_dir()):
        for image_path in sorted(identity_dir.iterdir()):
            if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                items.append((identity_dir.name, image_path))

    if len(items) < 2:
        raise RuntimeError(f"Need at least two test images under {test_dir}.")
    return items


def _load_resized_face(image_path: Path) -> np.ndarray:
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise ValueError(f"Failed to read image: {image_path}")
    resized_bgr = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)


def _count_images(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(
        1
        for image_path in path.rglob("*")
        if image_path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def _write_history_csv(path: Path, history: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(history)


def _write_metrics_csv(path: Path, metrics: dict[str, Any]) -> None:
    keys = [
        "pair_count",
        "positive_pair_count",
        "negative_pair_count",
        "verification_accuracy",
        "best_threshold",
        "auc",
        "tpr_at_best_threshold",
        "fpr_at_best_threshold",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        for key in keys:
            writer.writerow([key, metrics[key]])


def _write_results_table(
    path: Path,
    architecture: str,
    metrics: dict[str, Any],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["model_name", "distance_metric", "auc", "verification_accuracy"])
        writer.writerow(
            [
                architecture,
                "cosine_similarity",
                metrics["auc"],
                metrics["verification_accuracy"],
            ]
        )


def _cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator == 0:
        return 0.0
    return float(np.dot(first, second) / denominator)


def _roc_curve(y_true: np.ndarray, y_scores: np.ndarray) -> list[dict[str, float]]:
    thresholds = np.r_[np.inf, np.sort(np.unique(y_scores))[::-1], -np.inf]
    points = []
    positives = max(int(np.sum(y_true == 1)), 1)
    negatives = max(int(np.sum(y_true == 0)), 1)
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        points.append(
            {
                "threshold": float(threshold) if np.isfinite(threshold) else str(threshold),
                "tpr": float(tp / positives),
                "fpr": float(fp / negatives),
            }
        )
    return points


def _auc(points: list[dict[str, float]]) -> float:
    numeric_points = sorted((point["fpr"], point["tpr"]) for point in points)
    area = 0.0
    for (prev_fpr, prev_tpr), (fpr, tpr) in zip(numeric_points, numeric_points[1:]):
        area += (fpr - prev_fpr) * (tpr + prev_tpr) / 2
    return area


def _best_threshold(y_true: np.ndarray, y_scores: np.ndarray) -> tuple[float, float]:
    thresholds = np.sort(np.unique(y_scores))
    best_threshold = float(thresholds[0])
    best_accuracy = -1.0
    for threshold in thresholds:
        predictions = (y_scores >= threshold).astype(int)
        accuracy = float(np.mean(predictions == y_true))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = float(threshold)
    return best_threshold, best_accuracy


def _plot_loss_curve(
    path: Path,
    history: list[dict[str, float]],
    architecture: str,
) -> None:
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, train_loss, label="Train Loss")
    ax.plot(epochs, val_loss, label="Validation Loss")
    ax.set_title(f"{architecture} Triplet Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Triplet Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_roc_curve(
    path: Path,
    roc_curve: list[dict[str, float]],
    auc_score: float,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    fpr = [float(point["fpr"]) for point in roc_curve]
    tpr = [float(point["tpr"]) for point in roc_curve]
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set_title(title)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
