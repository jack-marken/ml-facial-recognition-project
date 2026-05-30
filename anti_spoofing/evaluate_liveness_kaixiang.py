import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from anti_spoofing.liveness_dataset_kaixiang import LivenessImageDataset
from anti_spoofing.liveness_models_kaixiang import load_checkpoint
from anti_spoofing.liveness_training_kaixiang import (
    calculate_metrics,
    make_transforms,
)


def try_roc_auc(labels, probabilities):
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(labels, probabilities))
    except Exception:
        return None


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint_path,
    data_dir,
    split,
    batch_size,
    workers,
    device_arg,
    threshold,
):
    device = torch.device(device_arg or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, checkpoint, device = load_checkpoint(checkpoint_path, device=device)

    dataset = LivenessImageDataset(
        data_dir, split=split, transform=make_transforms(train=False)
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )

    criterion = nn.BCEWithLogitsLoss()
    labels = []
    probabilities = []
    running_loss = 0.0

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    for images, targets in loader:
        images = images.to(device)
        targets = targets.float().to(device)

        logits = model(images).squeeze(1)
        loss = criterion(logits, targets)
        probs = torch.sigmoid(logits)

        running_loss += loss.item() * images.size(0)
        labels.extend(targets.cpu().int().tolist())
        probabilities.extend(probs.cpu().tolist())

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    metrics = calculate_metrics(labels, probabilities, threshold=threshold)
    metrics["loss"] = running_loss / max(1, len(dataset))
    metrics["roc_auc"] = try_roc_auc(labels, probabilities)
    metrics["fps"] = len(dataset) / max(1e-12, elapsed)
    metrics["samples"] = len(dataset)
    metrics["split"] = split
    metrics["threshold"] = threshold
    metrics["checkpoint"] = str(checkpoint_path)
    metrics["model_name"] = checkpoint["model_name"]
    metrics["confusion_matrix"] = {
        "rows": ["true_real", "true_spoof"],
        "cols": ["pred_real", "pred_spoof"],
        "values": [
            [metrics["tp"], metrics["fn"]],
            [metrics["fp"], metrics["tn"]],
        ],
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a Kaixiang liveness checkpoint on val/test data."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Path to a trained liveness checkpoint.",
    )
    parser.add_argument("--data-dir", default="datasets/liveness")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Minimum REAL probability required to predict REAL.",
    )
    parser.add_argument("--save-json", type=Path, default=None)
    args = parser.parse_args()

    metrics = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        split=args.split,
        batch_size=args.batch_size,
        workers=args.workers,
        device_arg=args.device,
        threshold=args.threshold,
    )

    print(json.dumps(metrics, indent=2))
    if args.save_json is not None:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"Saved metrics to: {args.save_json}")


if __name__ == "__main__":
    main()
