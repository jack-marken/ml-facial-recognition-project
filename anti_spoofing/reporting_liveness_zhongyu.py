"""Report artifact helpers for Zhongyu's liveness training runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def save_liveness_report(
    *,
    model,
    history,
    args,
    output_path: Path,
    data_dir: Path,
    report_root: Path,
    test_dataset,
) -> None:
    """Save configuration, training curves, and final binary metrics."""
    architecture = str(args.architecture).lower()
    report_dir = report_root / architecture
    report_dir.mkdir(parents=True, exist_ok=True)

    sample_counts = {
        split: _count_images(data_dir / split) for split in ("train", "val", "test")
    }
    config = {
        "model_name": f"liveness_{architecture}",
        "architecture": architecture,
        "input_image_size": [224, 224, 3],
        "optimizer": "Adam",
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "loss_function": "binary_crossentropy",
        "metrics": ["accuracy", "precision", "recall"],
        "dataset_name": str(data_dir),
        "sample_counts": sample_counts,
        "data_augmentation": "none",
        "train_base": bool(args.train_base),
        "model_output": str(output_path),
    }
    _write_json(report_dir / "config.json", config)

    history_dict = {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }
    _write_json(report_dir / "training_history.json", history_dict)
    _write_history_csv(report_dir / "training_history.csv", history_dict)
    _plot_liveness_curves(report_dir / "training_curves.png", history_dict)

    y_true, y_scores = _collect_binary_predictions(model, test_dataset)
    test_metrics = _binary_metrics(y_true, y_scores, threshold=0.5)
    _write_json(report_dir / "test_metrics.json", test_metrics)
    _write_metrics_csv(report_dir / "test_metrics.csv", test_metrics)
    _plot_confusion_matrix(
        report_dir / "confusion_matrix.png",
        test_metrics["confusion_matrix"],
        class_names=["spoof", "real"],
    )
    _plot_roc_curve(
        report_dir / "roc_curve.png",
        test_metrics["roc_curve"],
        auc_score=test_metrics["auc"],
    )


def _count_images(path: Path) -> dict[str, int] | int:
    if not path.exists():
        return 0
    split_counts = {}
    for class_dir in sorted(child for child in path.iterdir() if child.is_dir()):
        split_counts[class_dir.name] = sum(
            1
            for image_path in class_dir.rglob("*")
            if image_path.suffix.lower() in IMAGE_EXTENSIONS
        )
    return split_counts


def _write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def _write_history_csv(path: Path, history: dict[str, list[float]]) -> None:
    keys = sorted(history)
    epoch_count = max((len(values) for values in history.values()), default=0)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", *keys])
        for index in range(epoch_count):
            writer.writerow(
                [
                    index + 1,
                    *[
                        history[key][index] if index < len(history[key]) else ""
                        for key in keys
                    ],
                ]
            )


def _write_metrics_csv(path: Path, metrics: dict[str, Any]) -> None:
    scalar_keys = [
        "test_accuracy",
        "precision",
        "recall",
        "f1_score",
        "auc",
        "threshold",
        "total_samples",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        for key in scalar_keys:
            writer.writerow([key, metrics[key]])


def _collect_binary_predictions(model, test_dataset) -> tuple[np.ndarray, np.ndarray]:
    y_true_batches = []
    y_score_batches = []
    for images, labels in test_dataset:
        predictions = model.predict(images, verbose=0).reshape(-1)
        y_true_batches.append(labels.numpy().reshape(-1))
        y_score_batches.append(predictions)
    return (
        np.concatenate(y_true_batches).astype(int),
        np.concatenate(y_score_batches).astype(float),
    )


def _binary_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    y_pred = (y_scores >= threshold).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    accuracy = (tp + tn) / max(len(y_true), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1_score = 2 * precision * recall / max(precision + recall, 1e-12)
    roc_curve = _roc_curve(y_true, y_scores)

    return {
        "test_accuracy": round(float(accuracy), 6),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1_score": round(float(f1_score), 6),
        "auc": round(float(_auc(roc_curve)), 6),
        "threshold": threshold,
        "total_samples": int(len(y_true)),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "roc_curve": roc_curve,
    }


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


def _plot_liveness_curves(path: Path, history: dict[str, list[float]]) -> None:
    import matplotlib.pyplot as plt

    epochs = range(1, len(history.get("loss", [])) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, history.get("loss", []), label="Train Loss")
    if "val_loss" in history:
        axes[0].plot(epochs, history["val_loss"], label="Validation Loss")
    axes[0].set_title("Liveness Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Binary Crossentropy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history.get("accuracy", []), label="Train Accuracy")
    if "val_accuracy" in history:
        axes[1].plot(epochs, history["val_accuracy"], label="Validation Accuracy")
    axes[1].set_title("Liveness Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_confusion_matrix(
    path: Path,
    confusion_matrix: list[list[int]],
    class_names: list[str],
) -> None:
    import matplotlib.pyplot as plt

    matrix = np.asarray(confusion_matrix)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    image = ax.imshow(matrix, cmap="Blues")
    ax.figure.colorbar(image, ax=ax)
    ax.set_xticks(range(len(class_names)), labels=class_names)
    ax.set_yticks(range(len(class_names)), labels=class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Liveness Confusion Matrix")

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_roc_curve(
    path: Path,
    roc_curve: list[dict[str, float]],
    auc_score: float,
) -> None:
    import matplotlib.pyplot as plt

    fpr = [float(point["fpr"]) for point in roc_curve]
    tpr = [float(point["tpr"]) for point in roc_curve]
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set_title("Liveness ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
