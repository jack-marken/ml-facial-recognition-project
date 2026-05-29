import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from face_verification.metric_learning.siamese_dataset_kaixiang import FixedPairDataset
from face_verification.metric_learning.siamese_models_kaixiang import (
    load_siamese_checkpoint,
    pairwise_distance,
)
from face_verification.metric_learning.siamese_training_kaixiang import (
    calculate_pair_metrics,
)


def main():
    parser = argparse.ArgumentParser(
        description="Plot ROC and distance-metric comparison for Kaixiang metric recognition."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/recognition_triplet_resnet18_kaixiang_final30b_best.pth"),
    )
    parser.add_argument("--data-dir", default="datasets/recognition")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-positive-pairs-per-identity", type=int, default=20)
    parser.add_argument("--max-negative-pairs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    labels, euclidean_distances, cosine_similarities = collect_scores(args)
    euclidean_scores = [-distance for distance in euclidean_distances]

    euclidean_metrics = calculate_pair_metrics(labels, euclidean_distances)
    cosine_metrics = calculate_similarity_metrics(labels, cosine_similarities)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.checkpoint.stem
    roc_path = args.output_dir / f"{stem}_{args.split}_roc_curve.png"
    metric_path = args.output_dir / f"{stem}_{args.split}_distance_metric_comparison.png"
    score_dist_path = args.output_dir / f"{stem}_{args.split}_score_distribution.png"

    plot_roc_curve(
        labels=labels,
        euclidean_scores=euclidean_scores,
        cosine_scores=cosine_similarities,
        euclidean_auc=euclidean_metrics["roc_auc"],
        cosine_auc=cosine_metrics["roc_auc"],
        output_path=roc_path,
    )
    plot_metric_comparison(
        euclidean_metrics=euclidean_metrics,
        cosine_metrics=cosine_metrics,
        output_path=metric_path,
    )
    plot_score_distribution(
        labels=labels,
        euclidean_distances=euclidean_distances,
        cosine_similarities=cosine_similarities,
        euclidean_threshold=euclidean_metrics["best_distance_threshold"],
        cosine_threshold=cosine_metrics["best_similarity_threshold"],
        output_path=score_dist_path,
    )

    print(f"Saved ROC curve: {roc_path}")
    print(f"Saved distance metric comparison: {metric_path}")
    print(f"Saved score distribution: {score_dist_path}")
    print("Euclidean:", euclidean_metrics)
    print("Cosine:", cosine_metrics)


@torch.no_grad()
def collect_scores(args):
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, _, _ = load_siamese_checkpoint(args.checkpoint, device=device)
    dataset = FixedPairDataset(
        Path(args.data_dir) / args.split,
        max_positive_pairs_per_identity=args.max_positive_pairs_per_identity,
        max_negative_pairs=args.max_negative_pairs,
        seed=args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )

    labels = []
    euclidean_distances = []
    cosine_similarities = []
    for first_images, second_images, targets in loader:
        first_images = first_images.to(device)
        second_images = second_images.to(device)

        first_embeddings, second_embeddings = model(first_images, second_images)
        distances = pairwise_distance(first_embeddings, second_embeddings)
        similarities = F.cosine_similarity(first_embeddings, second_embeddings)

        labels.extend(targets.tolist())
        euclidean_distances.extend(distances.cpu().tolist())
        cosine_similarities.extend(similarities.cpu().tolist())

    return labels, euclidean_distances, cosine_similarities


def calculate_similarity_metrics(labels, similarities):
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    similarities_tensor = torch.tensor(similarities, dtype=torch.float32)

    best_accuracy = 0.0
    best_threshold = 0.0
    for threshold in torch.unique(similarities_tensor).tolist():
        predictions = (similarities_tensor >= threshold).float()
        accuracy = float((predictions == labels_tensor).float().mean().item())
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = float(threshold)

    from sklearn.metrics import roc_auc_score

    return {
        "accuracy": best_accuracy,
        "best_similarity_threshold": best_threshold,
        "roc_auc": float(roc_auc_score(labels, similarities)),
    }


def plot_roc_curve(labels, euclidean_scores, cosine_scores, euclidean_auc, cosine_auc, output_path):
    from sklearn.metrics import roc_curve

    euclidean_fpr, euclidean_tpr, _ = roc_curve(labels, euclidean_scores)
    cosine_fpr, cosine_tpr, _ = roc_curve(labels, cosine_scores)

    plt.figure(figsize=(7, 5))
    plt.plot(
        euclidean_fpr,
        euclidean_tpr,
        label=f"Euclidean score (-distance), AUC={euclidean_auc:.4f}",
        linewidth=2,
    )
    plt.plot(
        cosine_fpr,
        cosine_tpr,
        label=f"Cosine similarity, AUC={cosine_auc:.4f}",
        linestyle="--",
        linewidth=2,
    )
    plt.plot([0, 1], [0, 1], color="gray", linestyle=":", label="Random baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Face Verification ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_metric_comparison(euclidean_metrics, cosine_metrics, output_path):
    metric_names = ["ROC-AUC", "Accuracy"]
    euclidean_values = [
        euclidean_metrics["roc_auc"],
        euclidean_metrics["accuracy"],
    ]
    cosine_values = [
        cosine_metrics["roc_auc"],
        cosine_metrics["accuracy"],
    ]
    x_positions = range(len(metric_names))
    bar_width = 0.35

    plt.figure(figsize=(7, 5))
    plt.bar(
        [position - bar_width / 2 for position in x_positions],
        euclidean_values,
        width=bar_width,
        label="Euclidean distance",
    )
    plt.bar(
        [position + bar_width / 2 for position in x_positions],
        cosine_values,
        width=bar_width,
        label="Cosine similarity",
    )
    plt.xticks(list(x_positions), metric_names)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Similarity Distance Metric Comparison")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    for index, value in enumerate(euclidean_values):
        plt.text(index - bar_width / 2, value + 0.01, f"{value:.4f}", ha="center")
    for index, value in enumerate(cosine_values):
        plt.text(index + bar_width / 2, value + 0.01, f"{value:.4f}", ha="center")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_score_distribution(
    labels,
    euclidean_distances,
    cosine_similarities,
    euclidean_threshold,
    cosine_threshold,
    output_path,
):
    positive_distances = [
        distance for label, distance in zip(labels, euclidean_distances) if int(label) == 1
    ]
    negative_distances = [
        distance for label, distance in zip(labels, euclidean_distances) if int(label) == 0
    ]
    positive_similarities = [
        similarity for label, similarity in zip(labels, cosine_similarities) if int(label) == 1
    ]
    negative_similarities = [
        similarity for label, similarity in zip(labels, cosine_similarities) if int(label) == 0
    ]

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(
        positive_distances,
        bins=40,
        alpha=0.65,
        label="Positive pairs (same person)",
        density=True,
    )
    axes[0].hist(
        negative_distances,
        bins=40,
        alpha=0.65,
        label="Negative pairs (different people)",
        density=True,
    )
    axes[0].axvline(
        euclidean_threshold,
        color="black",
        linestyle="--",
        label=f"Best threshold={euclidean_threshold:.4f}",
    )
    axes[0].set_title("Euclidean Distance Distribution")
    axes[0].set_xlabel("Euclidean distance")
    axes[0].set_ylabel("Density")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].hist(
        positive_similarities,
        bins=40,
        alpha=0.65,
        label="Positive pairs (same person)",
        density=True,
    )
    axes[1].hist(
        negative_similarities,
        bins=40,
        alpha=0.65,
        label="Negative pairs (different people)",
        density=True,
    )
    axes[1].axvline(
        cosine_threshold,
        color="black",
        linestyle="--",
        label=f"Best threshold={cosine_threshold:.4f}",
    )
    axes[1].set_title("Cosine Similarity Distribution")
    axes[1].set_xlabel("Cosine similarity")
    axes[1].set_ylabel("Density")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    figure.suptitle("Positive vs Negative Pair Score Distribution")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


if __name__ == "__main__":
    main()
