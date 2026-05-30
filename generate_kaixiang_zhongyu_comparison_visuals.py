import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RECOGNITION_MODELS = [
    {
        "name": "kaixiang_contrastive_resnet18",
        "owner": "Kaixiang",
        "kind": "kaixiang",
        "checkpoint": Path("models/recognition_siamese_resnet18_kaixiang_final2_best.pth"),
    },
    {
        "name": "kaixiang_contrastive_mobilenetv2",
        "owner": "Kaixiang",
        "kind": "kaixiang",
        "checkpoint": Path("models/recognition_siamese_mobilenetv2_kaixiang_final1_best.pth"),
    },
    {
        "name": "kaixiang_triplet_resnet18",
        "owner": "Kaixiang",
        "kind": "kaixiang",
        "checkpoint": Path("models/recognition_triplet_resnet18_kaixiang_final30b_best.pth"),
    },
    {
        "name": "kaixiang_triplet_mobilenetv2",
        "owner": "Kaixiang",
        "kind": "kaixiang",
        "checkpoint": Path("models/recognition_triplet_mobilenetv2_kaixiang_final30_best.pth"),
    },
    {
        "name": "zhongyu_triplet_resnet34",
        "owner": "Zhongyu",
        "kind": "zhongyu",
        "architecture": "resnet34",
        "checkpoint": Path("models/recognition_triplet_resnet34_zhongyu.pth"),
    },
    {
        "name": "zhongyu_triplet_efficientnet_b0",
        "owner": "Zhongyu",
        "kind": "zhongyu",
        "architecture": "efficientnet_b0",
        "checkpoint": Path("models/recognition_triplet_efficientnet_zhongyu.pth"),
    },
]


LIVENESS_MODELS = [
    {
        "name": "kaixiang_mobilenetv2",
        "owner": "Kaixiang",
        "framework": "torch",
        "checkpoint": Path("models/liveness_mobilenetv2_kaixiang_final1_best.pth"),
    },
    {
        "name": "kaixiang_efficientnetb0",
        "owner": "Kaixiang",
        "framework": "torch",
        "checkpoint": Path("models/liveness_efficientnetb0_kaixiang_final1_best.pth"),
    },
    {
        "name": "zhongyu_densenet121",
        "owner": "Zhongyu",
        "framework": "keras",
        "checkpoint": Path("models/liveness_densenet121_zhongyu.keras"),
    },
    {
        "name": "zhongyu_resnet50v2",
        "owner": "Zhongyu",
        "framework": "keras",
        "checkpoint": Path("models/liveness_resnet50v2_zhongyu.weights.h5"),
    },
]


def main():
    parser = argparse.ArgumentParser(
        description="Recompute Kaixiang vs Zhongyu recognition/liveness comparisons and report visuals."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reports/kaixiang_zhongyu_recomparison"))
    parser.add_argument("--recognition-data-dir", type=Path, default=Path("datasets/recognition"))
    parser.add_argument("--recognition-split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--recognition-batch-size", type=int, default=32)
    parser.add_argument("--metric-max-positive-pairs-per-identity", type=int, default=20)
    parser.add_argument("--metric-max-negative-pairs", type=int, default=1000)
    parser.add_argument("--metric-seed", type=int, default=42)
    parser.add_argument("--liveness-data-dir", type=Path, default=Path("datasets/liveness"))
    parser.add_argument("--liveness-split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--liveness-batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    recognition = evaluate_recognition(args)
    liveness = evaluate_liveness(args)

    output = {
        "recognition": recognition["summary"],
        "liveness": liveness["summary"],
    }
    json_path = args.output_dir / "comparison_results.json"
    json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    write_csv(args.output_dir / "recognition_comparison.csv", recognition["summary"])
    write_csv(args.output_dir / "liveness_comparison.csv", liveness["summary"])

    plot_recognition_visuals(recognition, args.output_dir)
    plot_liveness_visuals(liveness, args.output_dir)
    write_markdown_summary(args.output_dir / "comparison_summary.md", output)

    print(f"Saved comparison JSON: {json_path}")
    print(f"Saved visuals to: {args.output_dir}")


def evaluate_recognition(args):
    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader

    from face_verification.metric_learning.embedding_model_zhongyu import FaceEmbeddingModel
    from face_verification.metric_learning.siamese_dataset_kaixiang import FixedPairDataset
    from face_verification.metric_learning.siamese_models_kaixiang import (
        load_siamese_checkpoint,
        pairwise_distance,
    )

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dataset = FixedPairDataset(
        args.recognition_data_dir / args.recognition_split,
        max_positive_pairs_per_identity=args.metric_max_positive_pairs_per_identity,
        max_negative_pairs=args.metric_max_negative_pairs,
        seed=args.metric_seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.recognition_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    results = []
    print(f"Recognition dataset: {args.recognition_data_dir / args.recognition_split}")
    print(f"Recognition pairs: {len(dataset)}")
    for config in RECOGNITION_MODELS:
        print(f"Evaluating recognition: {config['name']}")
        ensure_exists(config["checkpoint"])
        if config["kind"] == "kaixiang":
            model, checkpoint, _ = load_siamese_checkpoint(config["checkpoint"], device=device)
            model_label = checkpoint.get("model_name", config["name"])
            embedding_fn = model.forward_once
        else:
            model = FaceEmbeddingModel(config["architecture"], pretrained=False)
            checkpoint = torch.load(config["checkpoint"], map_location=device, weights_only=False)
            model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
            model.to(device)
            model.eval()
            model_label = config["architecture"]
            embedding_fn = model

        labels, distances, similarities, elapsed = collect_recognition_scores(
            loader,
            embedding_fn,
            device,
            torch,
            F,
            pairwise_distance,
        )
        euclidean_scores = [-distance for distance in distances]
        euclidean = binary_score_metrics(labels, euclidean_scores, higher_is_positive=True)
        cosine = binary_score_metrics(labels, similarities, higher_is_positive=True)

        results.append(
            {
                "name": config["name"],
                "owner": config["owner"],
                "model_label": model_label,
                "checkpoint": str(config["checkpoint"]),
                "pairs": len(labels),
                "fps": len(labels) / max(elapsed, 1e-12),
                "euclidean": {
                    **euclidean,
                    "best_distance_threshold": -euclidean["best_score_threshold"],
                },
                "cosine": {
                    **cosine,
                    "best_similarity_threshold": cosine["best_score_threshold"],
                },
                "labels": labels,
                "euclidean_distances": distances,
                "cosine_similarities": similarities,
            }
        )

    summary = [strip_arrays(result) for result in results]
    return {"results": results, "summary": summary}


def collect_recognition_scores(loader, embedding_fn, device, torch, F, pairwise_distance):
    labels = []
    distances = []
    similarities = []
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for first_images, second_images, targets in loader:
            first_images = first_images.to(device)
            second_images = second_images.to(device)
            first_embeddings = embedding_fn(first_images)
            second_embeddings = embedding_fn(second_images)
            distances.extend(pairwise_distance(first_embeddings, second_embeddings).cpu().tolist())
            similarities.extend(F.cosine_similarity(first_embeddings, second_embeddings).cpu().tolist())
            labels.extend([int(label) for label in targets.tolist()])
    if device.type == "cuda":
        torch.cuda.synchronize()
    return labels, distances, similarities, time.perf_counter() - start


def evaluate_liveness(args):
    samples = collect_liveness_samples(args.liveness_data_dir, args.liveness_split)
    results = []

    print(f"Liveness dataset: {args.liveness_data_dir / args.liveness_split}")
    print(f"Liveness samples: {len(samples)}")
    for config in LIVENESS_MODELS:
        print(f"Evaluating liveness: {config['name']}")
        ensure_exists(config["checkpoint"])
        if config["framework"] == "torch":
            labels, probabilities, elapsed = collect_torch_liveness_scores(config, args)
        else:
            labels, probabilities, elapsed = collect_keras_liveness_scores(config, samples, args)

        metrics = binary_score_metrics(labels, probabilities, higher_is_positive=True)
        results.append(
            {
                "name": config["name"],
                "owner": config["owner"],
                "framework": config["framework"],
                "checkpoint": str(config["checkpoint"]),
                "samples": len(labels),
                "fps": len(labels) / max(elapsed, 1e-12),
                "roc_auc": metrics["roc_auc"],
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "best_threshold": metrics["best_score_threshold"],
                "tp": metrics["tp"],
                "tn": metrics["tn"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "labels": labels,
                "real_probabilities": probabilities,
            }
        )

    summary = [strip_arrays(result) for result in results]
    return {"results": results, "summary": summary}


def collect_torch_liveness_scores(config, args):
    import torch
    from torch.utils.data import DataLoader

    from anti_spoofing.liveness_dataset_kaixiang import LivenessImageDataset
    from anti_spoofing.liveness_models_kaixiang import load_checkpoint
    from anti_spoofing.liveness_training_kaixiang import make_transforms

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, _, device = load_checkpoint(config["checkpoint"], device=device)
    dataset = LivenessImageDataset(args.liveness_data_dir, args.liveness_split, transform=make_transforms(train=False))
    loader = DataLoader(
        dataset,
        batch_size=args.liveness_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    labels = []
    probabilities = []
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            logits = model(images).squeeze(1)
            probabilities.extend(torch.sigmoid(logits).detach().cpu().tolist())
            labels.extend([int(label) for label in targets.tolist()])
    if device.type == "cuda":
        torch.cuda.synchronize()
    return labels, probabilities, time.perf_counter() - start


def collect_keras_liveness_scores(config, samples, args):
    from PIL import Image

    from anti_spoofing.liveness_zhongyu import _load_model

    model = _load_model(config["checkpoint"])
    labels = []
    probabilities = []
    start = time.perf_counter()
    for start_index in range(0, len(samples), args.liveness_batch_size):
        batch_samples = samples[start_index : start_index + args.liveness_batch_size]
        batch_images = []
        for path, label in batch_samples:
            image = Image.open(path).convert("RGB").resize((224, 224))
            batch_images.append(np.asarray(image, dtype=np.float32))
            labels.append(int(label))
        batch = np.stack(batch_images, axis=0)
        probabilities.extend(np.clip(model.predict(batch, verbose=0).reshape(-1), 0.0, 1.0).astype(float).tolist())
    return labels, probabilities, time.perf_counter() - start


def collect_liveness_samples(data_dir, split):
    from anti_spoofing.liveness_dataset_kaixiang import IMAGE_EXTENSIONS, LABEL_TO_INDEX

    samples = []
    split_dir = Path(data_dir) / split
    for class_name, label in LABEL_TO_INDEX.items():
        class_dir = split_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing liveness folder: {class_dir}")
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((path, label))
    if not samples:
        raise RuntimeError(f"No liveness samples found in {split_dir}")
    return samples


def binary_score_metrics(labels, scores, higher_is_positive):
    from sklearn.metrics import auc, confusion_matrix, precision_recall_fscore_support, roc_auc_score, roc_curve

    labels_array = np.asarray(labels, dtype=np.int32)
    scores_array = np.asarray(scores, dtype=np.float64)
    ranking_scores = scores_array if higher_is_positive else -scores_array
    roc_auc = float(roc_auc_score(labels_array, ranking_scores))
    fpr, tpr, thresholds = roc_curve(labels_array, ranking_scores)

    best = None
    for threshold in thresholds:
        predictions = (ranking_scores >= threshold).astype(np.int32)
        accuracy = float(np.mean(predictions == labels_array))
        tn, fp, fn, tp = confusion_matrix(labels_array, predictions, labels=[0, 1]).ravel()
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels_array,
            predictions,
            average="binary",
            zero_division=0,
        )
        balanced_accuracy = 0.5 * (
            (tp / max(tp + fn, 1)) + (tn / max(tn + fp, 1))
        )
        candidate = {
            "accuracy": accuracy,
            "balanced_accuracy": float(balanced_accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "best_score_threshold": float(threshold),
        }
        if best is None or candidate["balanced_accuracy"] > best["balanced_accuracy"]:
            best = candidate

    return {
        **best,
        "roc_auc": roc_auc,
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "roc_auc_check": float(auc(fpr, tpr)),
    }


def plot_recognition_visuals(recognition, output_dir):
    results = recognition["results"]
    labels = [short_name(result["name"]) for result in results]
    aucs = [result["euclidean"]["roc_auc"] for result in results]
    accuracies = [result["euclidean"]["accuracy"] for result in results]
    fps_values = [result["fps"] for result in results]

    plot_metric_bar_with_fps(
        labels,
        aucs,
        accuracies,
        fps_values,
        output_dir / "recognition_model_auc_accuracy_fps.png",
        "Metric Recognition Model Comparison",
    )

    plt.figure(figsize=(8, 6))
    for result in results:
        plt.plot(
            result["euclidean"]["fpr"],
            result["euclidean"]["tpr"],
            linewidth=2,
            label=f"{short_name(result['name'])} AUC={result['euclidean']['roc_auc']:.3f}",
        )
    plt.plot([0, 1], [0, 1], linestyle=":", color="gray", label="Random baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Face Verification ROC Curves")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "recognition_roc_curves.png", dpi=200)
    plt.close()

    best = max(results, key=lambda result: result["euclidean"]["roc_auc"])
    plot_recognition_distance_metric_comparison(best, output_dir / "recognition_best_model_distance_metric_comparison.png")
    plot_recognition_score_distribution(best, output_dir / "recognition_best_model_score_distribution.png")


def plot_liveness_visuals(liveness, output_dir):
    results = liveness["results"]
    labels = [short_name(result["name"]) for result in results]
    aucs = [result["roc_auc"] for result in results]
    accuracies = [result["accuracy"] for result in results]
    fps_values = [result["fps"] for result in results]

    plot_metric_bar_with_fps(
        labels,
        aucs,
        accuracies,
        fps_values,
        output_dir / "liveness_model_auc_accuracy_fps.png",
        "Anti-Spoofing / Liveness Model Comparison",
    )

    plt.figure(figsize=(8, 6))
    for result in results:
        metrics = binary_score_metrics(result["labels"], result["real_probabilities"], higher_is_positive=True)
        plt.plot(
            metrics["fpr"],
            metrics["tpr"],
            linewidth=2,
            label=f"{short_name(result['name'])} AUC={metrics['roc_auc']:.3f}",
        )
    plt.plot([0, 1], [0, 1], linestyle=":", color="gray", label="Random baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Liveness Detection ROC Curves")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "liveness_roc_curves.png", dpi=200)
    plt.close()

    best = max(results, key=lambda result: result["roc_auc"])
    plot_liveness_score_distribution(best, output_dir / "liveness_best_model_score_distribution.png")


def plot_metric_bar_with_fps(labels, aucs, accuracies, fps_values, output_path, title):
    x_positions = np.arange(len(labels))
    width = 0.35
    figure, left_axis = plt.subplots(figsize=(12, 6))
    left_axis.bar(x_positions - width / 2, aucs, width=width, label="ROC-AUC")
    left_axis.bar(x_positions + width / 2, accuracies, width=width, label="Accuracy")
    left_axis.set_ylim(0, 1.05)
    left_axis.set_ylabel("Score")
    left_axis.set_xticks(x_positions)
    left_axis.set_xticklabels(labels, rotation=25, ha="right")
    left_axis.grid(axis="y", alpha=0.3)

    right_axis = left_axis.twinx()
    right_axis.plot(x_positions, fps_values, marker="o", color="black", label="FPS")
    right_axis.set_ylabel("FPS")

    handles_left, labels_left = left_axis.get_legend_handles_labels()
    handles_right, labels_right = right_axis.get_legend_handles_labels()
    left_axis.legend(handles_left + handles_right, labels_left + labels_right, loc="lower right")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_recognition_distance_metric_comparison(result, output_path):
    metrics = ["ROC-AUC", "Accuracy"]
    euclidean = [result["euclidean"]["roc_auc"], result["euclidean"]["accuracy"]]
    cosine = [result["cosine"]["roc_auc"], result["cosine"]["accuracy"]]
    x_positions = np.arange(len(metrics))
    width = 0.35
    plt.figure(figsize=(7, 5))
    plt.bar(x_positions - width / 2, euclidean, width=width, label="Euclidean distance")
    plt.bar(x_positions + width / 2, cosine, width=width, label="Cosine similarity")
    plt.xticks(x_positions, metrics)
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title(f"Distance Metric Comparison: {short_name(result['name'])}")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_recognition_score_distribution(result, output_path):
    labels = np.asarray(result["labels"])
    distances = np.asarray(result["euclidean_distances"])
    similarities = np.asarray(result["cosine_similarities"])
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(distances[labels == 1], bins=40, alpha=0.65, density=True, label="Positive pairs")
    axes[0].hist(distances[labels == 0], bins=40, alpha=0.65, density=True, label="Negative pairs")
    axes[0].axvline(result["euclidean"]["best_distance_threshold"], color="black", linestyle="--", label="Best threshold")
    axes[0].set_title("Euclidean Distance Distribution")
    axes[0].set_xlabel("Euclidean distance")
    axes[0].set_ylabel("Density")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].hist(similarities[labels == 1], bins=40, alpha=0.65, density=True, label="Positive pairs")
    axes[1].hist(similarities[labels == 0], bins=40, alpha=0.65, density=True, label="Negative pairs")
    axes[1].axvline(result["cosine"]["best_similarity_threshold"], color="black", linestyle="--", label="Best threshold")
    axes[1].set_title("Cosine Similarity Distribution")
    axes[1].set_xlabel("Cosine similarity")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    figure.suptitle(f"Face Verification Score Distribution: {short_name(result['name'])}")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_liveness_score_distribution(result, output_path):
    labels = np.asarray(result["labels"])
    probabilities = np.asarray(result["real_probabilities"])
    plt.figure(figsize=(8, 5))
    plt.hist(probabilities[labels == 1], bins=40, alpha=0.65, density=True, label="Real faces")
    plt.hist(probabilities[labels == 0], bins=40, alpha=0.65, density=True, label="Spoof faces")
    plt.axvline(result["best_threshold"], color="black", linestyle="--", label=f"Best threshold={result['best_threshold']:.3f}")
    plt.xlabel("Predicted REAL probability")
    plt.ylabel("Density")
    plt.title(f"Liveness Score Distribution: {short_name(result['name'])}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys() if not isinstance(row.get(key), (list, dict))})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_markdown_summary(path, output):
    best_liveness = max(output["liveness"], key=lambda row: row["roc_auc"])
    best_recognition = max(output["recognition"], key=lambda row: row["euclidean"]["roc_auc"])
    lines = [
        "# Kaixiang vs Zhongyu Recomparison",
        "",
        "This report recomputes all models on the same local dataset splits and generates ROC, AUC, threshold, and score-distribution visuals.",
        "",
        f"Best liveness model by ROC-AUC: **{best_liveness['name']}** ({best_liveness['roc_auc']:.4f}).",
        f"Best recognition model by ROC-AUC: **{best_recognition['name']}** ({best_recognition['euclidean']['roc_auc']:.4f}).",
        "",
        "## Output Figures",
        "",
        "- `recognition_model_auc_accuracy_fps.png`",
        "- `recognition_roc_curves.png`",
        "- `recognition_best_model_distance_metric_comparison.png`",
        "- `recognition_best_model_score_distribution.png`",
        "- `liveness_model_auc_accuracy_fps.png`",
        "- `liveness_roc_curves.png`",
        "- `liveness_best_model_score_distribution.png`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def strip_arrays(result):
    return {
        key: value
        for key, value in result.items()
        if key not in {"labels", "euclidean_distances", "cosine_similarities", "real_probabilities"}
    }


def short_name(name):
    return (
        name.replace("kaixiang_", "kx_")
        .replace("zhongyu_", "zy_")
        .replace("contrastive_", "cont_")
        .replace("triplet_", "tri_")
    )


def ensure_exists(path):
    if not Path(path).exists():
        raise FileNotFoundError(f"Missing file: {path}")


if __name__ == "__main__":
    main()
