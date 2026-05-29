import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from glasses_detection.dataset_kaixiang import GlassesDataset
from glasses_detection.models_kaixiang import load_glasses_checkpoint
from glasses_detection.training_kaixiang import calculate_binary_metrics


@torch.no_grad()
def evaluate_checkpoint(checkpoint_path, data_dir, split, batch_size, workers, device=None):
    model, checkpoint, selected_device = load_glasses_checkpoint(checkpoint_path, device=device)
    dataset = GlassesDataset(Path(data_dir) / split, train=False)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )
    loss_function = nn.BCEWithLogitsLoss()
    all_labels = []
    all_logits = []
    total_loss = 0.0
    total_samples = 0
    start_time = time.perf_counter()
    for images, labels in loader:
        images = images.to(selected_device)
        labels = labels.to(selected_device)
        logits = model(images)
        loss = loss_function(logits, labels)
        total_loss += float(loss.item()) * images.size(0)
        total_samples += images.size(0)
        all_logits.extend(logits.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    metrics = calculate_binary_metrics(all_labels, all_logits)
    metrics["loss"] = total_loss / max(total_samples, 1)
    metrics["fps"] = total_samples / max(time.perf_counter() - start_time, 1e-12)
    metrics["samples"] = total_samples
    metrics["split"] = split
    metrics["checkpoint"] = str(Path(checkpoint_path))
    metrics["model_name"] = checkpoint["model_name"]
    metrics["confusion_matrix"] = {
        "rows": ["true_without_glasses", "true_with_glasses"],
        "cols": ["pred_without_glasses", "pred_with_glasses"],
        "values": [
            [metrics["tn"], metrics["fp"]],
            [metrics["fn"], metrics["tp"]],
        ],
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate a Kaixiang glasses detection checkpoint.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-dir", default="datasets/glasses")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-json", type=Path, default=None)
    args = parser.parse_args()

    metrics = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        split=args.split,
        batch_size=args.batch_size,
        workers=args.workers,
        device=args.device,
    )
    print(json.dumps(metrics, indent=2))
    if args.save_json:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"Saved metrics: {args.save_json}")


if __name__ == "__main__":
    main()

