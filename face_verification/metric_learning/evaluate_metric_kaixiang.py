import argparse
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from face_verification.metric_learning.siamese_dataset_kaixiang import FixedPairDataset
from face_verification.metric_learning.siamese_models_kaixiang import (
    load_siamese_checkpoint,
    pairwise_distance,
)
from face_verification.metric_learning.siamese_training_kaixiang import (
    ContrastiveLoss,
    calculate_pair_metrics,
    try_roc_auc,
)


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint_path,
    data_dir,
    split,
    batch_size,
    workers,
    device_arg,
    max_positive_pairs_per_identity,
    max_negative_pairs,
    seed,
    distance_threshold,
):
    device = torch.device(device_arg or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, checkpoint, device = load_siamese_checkpoint(checkpoint_path, device=device)
    dataset = FixedPairDataset(
        Path(data_dir) / split,
        max_positive_pairs_per_identity=max_positive_pairs_per_identity,
        max_negative_pairs=max_negative_pairs,
        seed=seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )
    loss_function = ContrastiveLoss(margin=checkpoint.get("args", {}).get("margin", 1.0))

    labels = []
    distances = []
    cosine_similarities = []
    total_loss = 0.0
    total_samples = 0

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    for first_images, second_images, targets in loader:
        first_images = first_images.to(device)
        second_images = second_images.to(device)
        targets = targets.to(device)

        first_embeddings, second_embeddings = model(first_images, second_images)
        batch_distances = pairwise_distance(first_embeddings, second_embeddings)
        batch_cosine_similarities = F.cosine_similarity(first_embeddings, second_embeddings)
        loss = loss_function(batch_distances, targets)

        batch_size_value = first_images.size(0)
        total_loss += float(loss.item()) * batch_size_value
        total_samples += batch_size_value
        labels.extend(targets.cpu().tolist())
        distances.extend(batch_distances.cpu().tolist())
        cosine_similarities.extend(batch_cosine_similarities.cpu().tolist())

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    metrics = calculate_pair_metrics(labels, distances)
    metrics["distance_metric"] = "euclidean"
    metrics["cosine"] = calculate_similarity_metrics(labels, cosine_similarities)
    if distance_threshold is not None:
        metrics["accuracy_at_requested_threshold"] = calculate_accuracy_at_threshold(
            labels,
            distances,
            distance_threshold,
        )
        metrics["requested_distance_threshold"] = distance_threshold

    metrics["loss"] = total_loss / max(total_samples, 1)
    metrics["fps"] = total_samples / max(1e-12, elapsed)
    metrics["pairs"] = total_samples
    metrics["split"] = split
    metrics["checkpoint"] = str(checkpoint_path)
    metrics["model_name"] = checkpoint["model_name"]
    return metrics


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

    return {
        "accuracy": best_accuracy,
        "best_similarity_threshold": best_threshold,
        "roc_auc": try_roc_auc(labels, similarities),
    }


def calculate_accuracy_at_threshold(labels, distances, threshold):
    correct = 0
    for label, distance in zip(labels, distances):
        prediction = 1.0 if distance <= threshold else 0.0
        correct += int(prediction == float(label))
    return correct / max(1, len(labels))


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Kaixiang Siamese metric-learning recognition."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-dir", default="datasets/recognition")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-positive-pairs-per-identity", type=int, default=20)
    parser.add_argument("--max-negative-pairs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--distance-threshold", type=float, default=None)
    parser.add_argument("--save-json", type=Path, default=None)
    args = parser.parse_args()

    metrics = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        split=args.split,
        batch_size=args.batch_size,
        workers=args.workers,
        device_arg=args.device,
        max_positive_pairs_per_identity=args.max_positive_pairs_per_identity,
        max_negative_pairs=args.max_negative_pairs,
        seed=args.seed,
        distance_threshold=args.distance_threshold,
    )
    print(json.dumps(metrics, indent=2))

    if args.save_json is not None:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"Saved metrics to: {args.save_json}")


if __name__ == "__main__":
    main()
