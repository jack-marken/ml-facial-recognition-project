import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from glasses_detection.dataset_kaixiang import GlassesDataset, count_split_images
from glasses_detection.models_kaixiang import (
    build_glasses_model,
    freeze_backbone,
    save_glasses_checkpoint,
    unfreeze_last_feature_blocks,
)


def add_training_args(parser, defaults):
    parser.add_argument("--data-dir", default="datasets/glasses")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--batch-size", type=int, default=defaults["batch_size"])
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--head-epochs", type=int, default=defaults["head_epochs"])
    parser.add_argument("--finetune-epochs", type=int, default=defaults["finetune_epochs"])
    parser.add_argument("--unfreeze-blocks", type=int, default=defaults["unfreeze_blocks"])
    parser.add_argument("--head-lr", type=float, default=defaults["head_lr"])
    parser.add_argument("--finetune-lr", type=float, default=defaults["finetune_lr"])
    parser.add_argument("--weight-decay", type=float, default=defaults["weight_decay"])
    parser.add_argument("--dropout", type=float, default=defaults["dropout"])
    parser.add_argument("--early-stopping-patience", type=int, default=defaults["early_stopping_patience"])
    parser.add_argument("--max-train-per-class", type=int, default=defaults["max_train_per_class"])
    parser.add_argument("--max-val-per-class", type=int, default=defaults["max_val_per_class"])
    parser.add_argument("--progress-every", type=int, default=defaults["progress_every"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-pretrained-backbone", action="store_true")


def run_glasses_training(model_name, args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    run_stem = f"glasses_{model_name}_kaixiang"
    if args.run_name:
        run_stem += f"_{args.run_name}"
    best_path = output_dir / f"{run_stem}_best.pth"
    latest_path = output_dir / f"{run_stem}_latest.pth"
    history_path = output_dir / f"{run_stem}_history.json"

    print(f"Training glasses model: {model_name}")
    print(f"Device: {device}")
    print(f"Dataset: {args.data_dir}")
    if args.run_name:
        print(f"Run name: {args.run_name}")
    print(
        "Hyperparameters: "
        f"batch_size={args.batch_size}, head_epochs={args.head_epochs}, "
        f"finetune_epochs={args.finetune_epochs}, unfreeze_blocks={args.unfreeze_blocks}, "
        f"head_lr={args.head_lr}, finetune_lr={args.finetune_lr}, "
        f"weight_decay={args.weight_decay}, early_stopping_patience={args.early_stopping_patience}"
    )

    train_dataset = GlassesDataset(
        Path(args.data_dir) / "train",
        train=True,
        max_per_class=args.max_train_per_class,
        seed=args.seed,
    )
    val_dataset = GlassesDataset(
        Path(args.data_dir) / "val",
        train=False,
        max_per_class=args.max_val_per_class,
        seed=args.seed,
    )
    print(f"Train samples used: {len(train_dataset)}")
    print(f"Val samples used: {len(val_dataset)}")
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

    counts = count_split_images(args.data_dir)
    positives = counts["train"]["with_glasses"]
    negatives = counts["train"]["without_glasses"]
    pos_weight_value = negatives / max(positives, 1)
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    print(f"Positive class weight: {pos_weight_value:.4f}")

    model = build_glasses_model(
        model_name,
        pretrained_backbone=not args.no_pretrained_backbone,
        dropout=args.dropout,
    )
    model.to(device)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    history = []
    args_dict = vars(args).copy()
    best_score = -1.0
    epoch_index = 0
    start_time = time.time()

    print("Stage 1: freeze backbone, train binary head")
    freeze_backbone(model)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.head_lr,
        weight_decay=args.weight_decay,
    )
    for _ in range(args.head_epochs):
        epoch_index += 1
        train_metrics = run_epoch(
            model,
            train_loader,
            loss_function,
            device,
            optimizer,
            progress_every=args.progress_every,
            stage_name=f"head epoch {epoch_index:03d}",
        )
        val_metrics = evaluate(
            model,
            val_loader,
            loss_function,
            device,
            progress_every=args.progress_every,
            stage_name=f"val epoch {epoch_index:03d}",
        )
        history.append({"epoch": epoch_index, "stage": "head", "train": train_metrics, "val": val_metrics})
        print_summary(epoch_index, "head", train_metrics, val_metrics)
        save_glasses_checkpoint(latest_path, model, model_name, epoch_index, val_metrics, args_dict)
        best_score = maybe_save_best(best_path, model, model_name, epoch_index, val_metrics, args_dict, best_score)

    print(f"Stage 2: unfreeze last {args.unfreeze_blocks} feature blocks and fine-tune")
    unfreeze_last_feature_blocks(model, model_name, args.unfreeze_blocks)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.finetune_lr,
        weight_decay=args.weight_decay,
    )
    epochs_without_improvement = 0
    for _ in range(args.finetune_epochs):
        epoch_index += 1
        previous_best = best_score
        train_metrics = run_epoch(
            model,
            train_loader,
            loss_function,
            device,
            optimizer,
            progress_every=args.progress_every,
            stage_name=f"fine_tune epoch {epoch_index:03d}",
        )
        val_metrics = evaluate(
            model,
            val_loader,
            loss_function,
            device,
            progress_every=args.progress_every,
            stage_name=f"val epoch {epoch_index:03d}",
        )
        history.append({"epoch": epoch_index, "stage": "fine_tune", "train": train_metrics, "val": val_metrics})
        print_summary(epoch_index, "fine_tune", train_metrics, val_metrics)
        save_glasses_checkpoint(latest_path, model, model_name, epoch_index, val_metrics, args_dict)
        best_score = maybe_save_best(best_path, model, model_name, epoch_index, val_metrics, args_dict, best_score)

        if best_score <= previous_best:
            epochs_without_improvement += 1
        else:
            epochs_without_improvement = 0
        if epochs_without_improvement > args.early_stopping_patience:
            print("Early stopping triggered: val_f1 did not improve.")
            break

    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print("Training complete.")
    print(f"Best checkpoint: {best_path}")
    print(f"Latest checkpoint: {latest_path}")
    print(f"History: {history_path}")
    print(f"Elapsed seconds: {time.time() - start_time:.1f}")


def run_epoch(model, loader, loss_function, device, optimizer, progress_every=100, stage_name="train"):
    model.train()
    total_loss = 0.0
    total_samples = 0
    all_logits = []
    all_labels = []
    for batch_index, (images, labels) in enumerate(loader, start=1):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_function(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * images.size(0)
        total_samples += images.size(0)
        all_logits.extend(logits.detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())
        if progress_every and batch_index % progress_every == 0:
            print(f"  {stage_name}: batch {batch_index}/{len(loader)} loss={loss.item():.4f}", flush=True)
    metrics = calculate_binary_metrics(all_labels, all_logits)
    metrics["loss"] = total_loss / max(total_samples, 1)
    return metrics


@torch.no_grad()
def evaluate(model, loader, loss_function, device, progress_every=100, stage_name="val"):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_logits = []
    all_labels = []
    start_time = time.perf_counter()
    for batch_index, (images, labels) in enumerate(loader, start=1):
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = loss_function(logits, labels)
        total_loss += float(loss.item()) * images.size(0)
        total_samples += images.size(0)
        all_logits.extend(logits.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        if progress_every and batch_index % progress_every == 0:
            print(f"  {stage_name}: batch {batch_index}/{len(loader)}", flush=True)
    metrics = calculate_binary_metrics(all_labels, all_logits)
    metrics["loss"] = total_loss / max(total_samples, 1)
    metrics["fps"] = total_samples / max(time.perf_counter() - start_time, 1e-12)
    return metrics


def calculate_binary_metrics(labels, logits, threshold=0.5):
    probabilities = torch.sigmoid(torch.tensor(logits, dtype=torch.float32))
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    predictions = (probabilities >= threshold).float()

    tp = int(((predictions == 1) & (labels_tensor == 1)).sum().item())
    tn = int(((predictions == 0) & (labels_tensor == 0)).sum().item())
    fp = int(((predictions == 1) & (labels_tensor == 0)).sum().item())
    fn = int(((predictions == 0) & (labels_tensor == 1)).sum().item())
    total = max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "accuracy": (tp + tn) / total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "threshold": threshold,
    }


def maybe_save_best(path, model, model_name, epoch, val_metrics, args_dict, best_score):
    score = val_metrics["f1"]
    if score > best_score:
        save_glasses_checkpoint(path, model, model_name, epoch, val_metrics, args_dict)
        return score
    return best_score


def print_summary(epoch, stage, train_metrics, val_metrics):
    print(
        f"Epoch {epoch:03d} [{stage}] "
        f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} "
        f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
        f"val_precision={val_metrics['precision']:.4f} val_recall={val_metrics['recall']:.4f} "
        f"val_f1={val_metrics['f1']:.4f}"
    )
