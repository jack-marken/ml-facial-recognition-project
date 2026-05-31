import argparse
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from face_verification.metric_learning.embedding_model_zhongyu import (
    FaceEmbeddingModel,
)
from face_verification.metric_learning.siamese_dataset_kaixiang import FixedPairDataset
from face_verification.metric_learning.siamese_models_kaixiang import (
    load_siamese_checkpoint,
    pairwise_distance,
)
from face_verification.metric_learning.siamese_training_kaixiang import (
    calculate_pair_metrics,
    try_roc_auc,
)


DEFAULT_MODELS = [
    {
        "name": "kaixiang_contrastive_resnet18",
        "owner": "kaixiang",
        "kind": "siamese",
        "checkpoint": "models/recognition_siamese_resnet18_kaixiang_final2_best.pth",
    },
    {
        "name": "kaixiang_contrastive_mobilenetv2",
        "owner": "kaixiang",
        "kind": "siamese",
        "checkpoint": "models/recognition_siamese_mobilenetv2_kaixiang_final1_best.pth",
    },
    {
        "name": "kaixiang_triplet_resnet18",
        "owner": "kaixiang",
        "kind": "siamese",
        "checkpoint": "models/recognition_triplet_resnet18_kaixiang_final30b_best.pth",
    },
    {
        "name": "kaixiang_triplet_mobilenetv2",
        "owner": "kaixiang",
        "kind": "siamese",
        "checkpoint": "models/recognition_triplet_mobilenetv2_kaixiang_final30_best.pth",
    },
    {
        "name": "zhongyu_triplet_resnet34",
        "owner": "zhongyu",
        "kind": "zhongyu_triplet",
        "architecture": "resnet34",
        "checkpoint": "models/recognition_triplet_resnet34_zhongyu.pth",
    },
    {
        "name": "zhongyu_triplet_efficientnet_b0",
        "owner": "zhongyu",
        "kind": "zhongyu_triplet",
        "architecture": "efficientnet_b0",
        "checkpoint": "models/recognition_triplet_efficientnet_zhongyu.pth",
    },
]


@torch.no_grad()
def evaluate_model(model_config, loader, device):
    checkpoint_path = Path(model_config["checkpoint"])
    if not checkpoint_path.exists():
        return {
            "name": model_config["name"],
            "owner": model_config["owner"],
            "error": f"checkpoint not found: {checkpoint_path}",
        }

    if model_config["kind"] == "siamese":
        model, checkpoint, _ = load_siamese_checkpoint(checkpoint_path, device=device)
        model_label = checkpoint.get("model_name", model_config["name"])
        embedding_function = model.forward_once
    else:
        model = load_zhongyu_triplet_model(model_config, checkpoint_path, device)
        model_label = model_config["architecture"]
        embedding_function = model

    labels = []
    euclidean_distances = []
    cosine_similarities = []
    total_samples = 0

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    for first_images, second_images, targets in loader:
        first_images = first_images.to(device)
        second_images = second_images.to(device)

        first_embeddings = embedding_function(first_images)
        second_embeddings = embedding_function(second_images)

        distances = pairwise_distance(first_embeddings, second_embeddings)
        similarities = F.cosine_similarity(first_embeddings, second_embeddings)

        batch_size = first_images.size(0)
        total_samples += batch_size
        labels.extend(targets.tolist())
        euclidean_distances.extend(distances.cpu().tolist())
        cosine_similarities.extend(similarities.cpu().tolist())

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    euclidean_metrics = calculate_pair_metrics(labels, euclidean_distances)
    cosine_metrics = calculate_similarity_metrics(labels, cosine_similarities)

    return {
        "name": model_config["name"],
        "owner": model_config["owner"],
        "model_label": model_label,
        "checkpoint": str(checkpoint_path),
        "euclidean": euclidean_metrics,
        "cosine": cosine_metrics,
        "fps": total_samples / max(elapsed, 1e-12),
        "pairs": total_samples,
    }


def load_zhongyu_triplet_model(model_config, checkpoint_path, device):
    model = FaceEmbeddingModel(
        architecture=model_config["architecture"],
        pretrained=False,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


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


def print_summary(results):
    print("\nSummary:")
    print(
        "model,euclidean_auc,euclidean_acc,euclidean_threshold,"
        "cosine_auc,cosine_acc,cosine_threshold,fps"
    )
    for result in results:
        if "error" in result:
            print(f"{result['name']},ERROR,{result['error']}")
            continue
        euclidean = result["euclidean"]
        cosine = result["cosine"]
        print(
            f"{result['name']},"
            f"{euclidean['roc_auc']},"
            f"{euclidean['accuracy']},"
            f"{euclidean['best_distance_threshold']},"
            f"{cosine['roc_auc']},"
            f"{cosine['accuracy']},"
            f"{cosine['best_similarity_threshold']},"
            f"{result['fps']}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare Kaixiang and Zhongyu metric-learning models on the same pairs."
    )
    parser.add_argument("--data-dir", default="datasets/recognition")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-positive-pairs-per-identity", type=int, default=20)
    parser.add_argument("--max-negative-pairs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-json", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

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

    print(f"Device: {device}")
    print(f"Dataset: {args.data_dir}\\{args.split}")
    print(f"Pairs: {len(dataset)}")

    results = []
    for model_config in DEFAULT_MODELS:
        print(f"Evaluating {model_config['name']}...")
        results.append(evaluate_model(model_config, loader, device))

    output = {
        "split": args.split,
        "pairs": len(dataset),
        "results": results,
    }
    print(json.dumps(output, indent=2))
    print_summary(results)

    if args.save_json is not None:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(f"Saved comparison to: {args.save_json}")


if __name__ == "__main__":
    main()
