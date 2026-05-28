import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from face_verification.metric_learning.siamese_dataset_kaixiang import (
    FixedPairDataset,
    SiamesePairDataset,
)
from face_verification.metric_learning.siamese_models_kaixiang import (
    pairwise_distance,
)


class ContrastiveLoss(nn.Module):
    """Contrastive loss with label 1=same identity, 0=different identity."""

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, distances: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        positive_loss = labels * distances.pow(2)
        negative_loss = (1.0 - labels) * torch.clamp(self.margin - distances, min=0.0).pow(2)
        return (positive_loss + negative_loss).mean()


def make_loaders(args):
    train_dataset = SiamesePairDataset(
        Path(args.data_dir) / "train",
        pairs_per_epoch=args.pairs_per_epoch,
        seed=args.seed,
    )
    val_dataset = FixedPairDataset(
        Path(args.data_dir) / "val",
        max_positive_pairs_per_identity=args.max_positive_pairs_per_identity,
        max_negative_pairs=args.max_negative_pairs,
        seed=args.seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def calculate_pair_metrics(labels, distances):
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    distances_tensor = torch.tensor(distances, dtype=torch.float32)

    best_accuracy = 0.0
    best_threshold = 0.0
    for threshold in torch.unique(distances_tensor).tolist():
        predictions = (distances_tensor <= threshold).float()
        accuracy = float((predictions == labels_tensor).float().mean().item())
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = float(threshold)

    roc_auc = try_roc_auc(labels, [-distance for distance in distances])
    return {
        "accuracy": best_accuracy,
        "best_distance_threshold": best_threshold,
        "roc_auc": roc_auc,
    }


def try_roc_auc(labels, scores):
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(labels, scores))
    except Exception:
        return None


def run_one_epoch(model, loader, loss_function, device, optimizer=None, epoch_label="train"):
    if optimizer is None:
        model.eval()
    else:
        model.train()

    total_loss = 0.0
    total_samples = 0
    labels = []
    distances = []

    progress = tqdm(loader, desc=epoch_label, leave=False)
    for first_images, second_images, targets in progress:
        first_images = first_images.to(device)
        second_images = second_images.to(device)
        targets = targets.to(device)

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(optimizer is not None):
            first_embeddings, second_embeddings = model(first_images, second_images)
            batch_distances = pairwise_distance(first_embeddings, second_embeddings)
            loss = loss_function(batch_distances, targets)

        if optimizer is not None:
            loss.backward()
            optimizer.step()

        batch_size = first_images.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        labels.extend(targets.detach().cpu().tolist())
        distances.extend(batch_distances.detach().cpu().tolist())
        progress.set_postfix(loss=loss.item())

    metrics = calculate_pair_metrics(labels, distances)
    metrics["loss"] = total_loss / max(total_samples, 1)
    return metrics


def save_checkpoint(path, model, model_name, epoch, metrics, args_dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": model_name,
            "epoch": epoch,
            "metrics": metrics,
            "args": args_dict,
            "label_map": {"different": 0, "same": 1},
            "model_state_dict": model.state_dict(),
        },
        path,
    )


def make_run_stem(model_name, run_name):
    base = f"recognition_siamese_{model_name}_kaixiang"
    if not run_name:
        return base
    safe_run_name = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in run_name.strip()
    )
    return f"{base}_{safe_run_name}" if safe_run_name else base


def run_siamese_training(
    model_name,
    args,
    build_model,
    freeze_backbone,
    unfreeze_backbone,
):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    run_stem = make_run_stem(model_name, args.run_name)
    best_path = output_dir / f"{run_stem}_best.pth"
    latest_path = output_dir / f"{run_stem}_latest.pth"
    history_path = output_dir / f"{run_stem}_history.json"
    args_dict = vars(args).copy()
    history = []
    best_score = -1.0

    print(f"Training model: {model_name}")
    print(f"Device: {device}")
    print(f"Dataset: {args.data_dir}")
    print(f"Run name: {args.run_name or 'default'}")
    print(
        "Hyperparameters: "
        f"batch_size={args.batch_size}, "
        f"head_epochs={args.head_epochs}, "
        f"finetune_epochs={args.finetune_epochs}, "
        f"unfreeze_blocks={args.unfreeze_blocks}, "
        f"head_lr={args.head_lr}, "
        f"finetune_lr={args.finetune_lr}, "
        f"margin={args.margin}, "
        f"weight_decay={args.weight_decay}"
    )
    print("Preparing datasets and model...", flush=True)

    train_loader, val_loader = make_loaders(args)
    model = build_model(pretrained_backbone=not args.no_pretrained_backbone)
    model.to(device)
    loss_function = ContrastiveLoss(margin=args.margin)

    freeze_backbone(model)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.head_lr,
        weight_decay=args.weight_decay,
    )

    start_time = time.time()
    epoch_index = 0

    for epoch in range(1, args.head_epochs + 1):
        epoch_index += 1
        train_metrics = run_one_epoch(
            model,
            train_loader,
            loss_function,
            device,
            optimizer=optimizer,
            epoch_label=f"head {epoch}/{args.head_epochs}",
        )
        val_metrics = run_one_epoch(
            model,
            val_loader,
            loss_function,
            device,
            optimizer=None,
            epoch_label="val",
        )
        history.append(
            {"epoch": epoch_index, "stage": "head", "train": train_metrics, "val": val_metrics}
        )
        print_epoch_summary(epoch_index, "head", train_metrics, val_metrics)
        save_checkpoint(latest_path, model, model_name, epoch_index, val_metrics, args_dict)

        score = val_metrics["roc_auc"] if val_metrics["roc_auc"] is not None else val_metrics["accuracy"]
        if score > best_score:
            best_score = score
            save_checkpoint(best_path, model, model_name, epoch_index, val_metrics, args_dict)

    print(f"Stage 2: unfreeze last {args.unfreeze_blocks} backbone blocks and fine-tune")
    unfreeze_backbone(model, args.unfreeze_blocks)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.finetune_lr,
        weight_decay=args.weight_decay,
    )

    epochs_without_improvement = 0
    for epoch in range(1, args.finetune_epochs + 1):
        epoch_index += 1
        train_metrics = run_one_epoch(
            model,
            train_loader,
            loss_function,
            device,
            optimizer=optimizer,
            epoch_label=f"fine {epoch}/{args.finetune_epochs}",
        )
        val_metrics = run_one_epoch(
            model,
            val_loader,
            loss_function,
            device,
            optimizer=None,
            epoch_label="val",
        )
        history.append(
            {
                "epoch": epoch_index,
                "stage": "fine_tune",
                "train": train_metrics,
                "val": val_metrics,
            }
        )
        print_epoch_summary(epoch_index, "fine_tune", train_metrics, val_metrics)
        save_checkpoint(latest_path, model, model_name, epoch_index, val_metrics, args_dict)

        score = val_metrics["roc_auc"] if val_metrics["roc_auc"] is not None else val_metrics["accuracy"]
        if score > best_score:
            best_score = score
            epochs_without_improvement = 0
            save_checkpoint(best_path, model, model_name, epoch_index, val_metrics, args_dict)
        else:
            epochs_without_improvement += 1

        if (
            not args.disable_early_stopping
            and args.early_stopping_patience > 0
            and epochs_without_improvement > args.early_stopping_patience
        ):
            print(
                "Early stopping triggered: validation score did not improve "
                f"for more than {args.early_stopping_patience} fine-tuning epochs."
            )
            break

    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print("Training complete.")
    print(f"Best checkpoint: {best_path}")
    print(f"Latest checkpoint: {latest_path}")
    print(f"History: {history_path}")
    print(f"Elapsed seconds: {time.time() - start_time:.1f}")


def print_epoch_summary(epoch, stage, train_metrics, val_metrics):
    train_auc = train_metrics["roc_auc"]
    val_auc = val_metrics["roc_auc"]
    print(
        f"Epoch {epoch:03d} [{stage}] "
        f"train_loss={train_metrics['loss']:.4f} "
        f"train_acc={train_metrics['accuracy']:.4f} "
        f"train_auc={train_auc if train_auc is not None else 'NA'} "
        f"val_loss={val_metrics['loss']:.4f} "
        f"val_acc={val_metrics['accuracy']:.4f} "
        f"val_auc={val_auc if val_auc is not None else 'NA'} "
        f"val_threshold={val_metrics['best_distance_threshold']:.4f}"
    )


def add_common_training_args(parser, defaults):
    parser.add_argument("--data-dir", default="datasets/recognition")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--batch-size", type=int, default=defaults["batch_size"])
    parser.add_argument("--workers", type=int, default=defaults["workers"])
    parser.add_argument("--head-epochs", type=int, default=defaults["head_epochs"])
    parser.add_argument("--finetune-epochs", type=int, default=defaults["finetune_epochs"])
    parser.add_argument("--unfreeze-blocks", type=int, default=defaults["unfreeze_blocks"])
    parser.add_argument("--head-lr", type=float, default=defaults["head_lr"])
    parser.add_argument("--finetune-lr", type=float, default=defaults["finetune_lr"])
    parser.add_argument("--weight-decay", type=float, default=defaults["weight_decay"])
    parser.add_argument("--margin", type=float, default=defaults["margin"])
    parser.add_argument("--pairs-per-epoch", type=int, default=defaults["pairs_per_epoch"])
    parser.add_argument(
        "--max-positive-pairs-per-identity",
        type=int,
        default=defaults["max_positive_pairs_per_identity"],
    )
    parser.add_argument("--max-negative-pairs", type=int, default=defaults["max_negative_pairs"])
    parser.add_argument("--early-stopping-patience", type=int, default=defaults["early_stopping_patience"])
    parser.add_argument("--disable-early-stopping", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-pretrained-backbone", action="store_true")
