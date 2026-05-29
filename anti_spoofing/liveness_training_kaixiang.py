import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from anti_spoofing.liveness_dataset_kaixiang import LivenessImageDataset


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def make_transforms(train: bool):
    if train:
        return transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.15,
                    hue=0.03,
                ),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def make_loaders(data_dir, batch_size, workers):
    train_dataset = LivenessImageDataset(
        data_dir, split="train", transform=make_transforms(train=True)
    )
    val_dataset = LivenessImageDataset(
        data_dir, split="val", transform=make_transforms(train=False)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def calculate_metrics(labels, probabilities, threshold=0.5):
    predictions = [1 if prob >= threshold else 0 for prob in probabilities]

    tp = sum(1 for y, p in zip(labels, predictions) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(labels, predictions) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(labels, predictions) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, predictions) if y == 1 and p == 0)

    total = max(1, len(labels))
    accuracy = (tp + tn) / total
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def train_one_epoch(model, loader, criterion, optimizer, device, epoch_label):
    model.train()
    running_loss = 0.0
    labels = []
    probabilities = []

    progress = tqdm(loader, desc=epoch_label, leave=False)
    for images, targets in progress:
        images = images.to(device)
        targets = targets.float().to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images).squeeze(1)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        probs = torch.sigmoid(logits).detach().cpu().tolist()
        probabilities.extend(probs)
        labels.extend(targets.cpu().int().tolist())
        progress.set_postfix(loss=loss.item())

    metrics = calculate_metrics(labels, probabilities)
    metrics["loss"] = running_loss / max(1, len(loader.dataset))
    return metrics


@torch.no_grad()
def evaluate_one_epoch(model, loader, criterion, device, epoch_label="val"):
    model.eval()
    running_loss = 0.0
    labels = []
    probabilities = []

    progress = tqdm(loader, desc=epoch_label, leave=False)
    for images, targets in progress:
        images = images.to(device)
        targets = targets.float().to(device)

        logits = model(images).squeeze(1)
        loss = criterion(logits, targets)

        running_loss += loss.item() * images.size(0)
        probs = torch.sigmoid(logits).detach().cpu().tolist()
        probabilities.extend(probs)
        labels.extend(targets.cpu().int().tolist())

    metrics = calculate_metrics(labels, probabilities)
    metrics["loss"] = running_loss / max(1, len(loader.dataset))
    return metrics


def save_checkpoint(path, model, model_name, epoch, metrics, args_dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": model_name,
            "epoch": epoch,
            "metrics": metrics,
            "args": args_dict,
            "label_map": {"spoof": 0, "real": 1},
            "model_state_dict": model.state_dict(),
        },
        path,
    )


def make_run_stem(model_name, run_name):
    base = f"liveness_{model_name}_kaixiang"
    if not run_name:
        return base

    safe_run_name = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in run_name.strip()
    )
    return f"{base}_{safe_run_name}" if safe_run_name else base


def run_liveness_training(
    model_name,
    args,
    build_model,
    freeze_backbone,
    unfreeze_last_blocks,
):
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_loader, val_loader = make_loaders(data_dir, args.batch_size, args.workers)

    model = build_model(pretrained_backbone=not args.no_pretrained_backbone)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    history = []
    best_f1 = -1.0
    run_stem = make_run_stem(model_name, args.run_name)
    best_path = output_dir / f"{run_stem}_best.pth"
    latest_path = output_dir / f"{run_stem}_latest.pth"
    args_dict = vars(args).copy()

    print(f"Training model: {model_name}")
    print(f"Device: {device}")
    print(f"Dataset: {data_dir}")
    print(f"Run name: {args.run_name or 'default'}")
    print(
        "Hyperparameters: "
        f"batch_size={args.batch_size}, "
        f"head_epochs={args.head_epochs}, "
        f"finetune_epochs={args.finetune_epochs}, "
        f"unfreeze_blocks={args.unfreeze_blocks}, "
        f"head_lr={args.head_lr}, "
        f"finetune_lr={args.finetune_lr}, "
        f"weight_decay={args.weight_decay}, "
        f"early_stopping_patience={args.early_stopping_patience}"
    )
    print("Stage 1: freeze backbone, train binary head")

    freeze_backbone(model)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.head_lr,
        weight_decay=args.weight_decay,
    )

    start_time = time.time()
    epoch_index = 0

    for epoch in range(1, args.head_epochs + 1):
        epoch_index += 1
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch_label=f"head {epoch}/{args.head_epochs}",
        )
        val_metrics = evaluate_one_epoch(
            model, val_loader, criterion, device, epoch_label="val"
        )
        history.append(
            {
                "epoch": epoch_index,
                "stage": "head",
                "train": train_metrics,
                "val": val_metrics,
            }
        )
        print_epoch_summary(epoch_index, "head", train_metrics, val_metrics)

        save_checkpoint(latest_path, model, model_name, epoch_index, val_metrics, args_dict)
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            save_checkpoint(best_path, model, model_name, epoch_index, val_metrics, args_dict)

    print(f"Stage 2: unfreeze last {args.unfreeze_blocks} feature blocks and fine-tune")
    unfreeze_last_blocks(model, args.unfreeze_blocks)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.finetune_lr,
        weight_decay=args.weight_decay,
    )

    fine_tune_epochs_without_improvement = 0
    for epoch in range(1, args.finetune_epochs + 1):
        epoch_index += 1
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch_label=f"fine {epoch}/{args.finetune_epochs}",
        )
        val_metrics = evaluate_one_epoch(
            model, val_loader, criterion, device, epoch_label="val"
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
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            fine_tune_epochs_without_improvement = 0
            save_checkpoint(best_path, model, model_name, epoch_index, val_metrics, args_dict)
        else:
            fine_tune_epochs_without_improvement += 1

        if (
            not args.disable_early_stopping
            and args.early_stopping_patience > 0
            and fine_tune_epochs_without_improvement > args.early_stopping_patience
        ):
            print(
                "Early stopping triggered: "
                f"val_f1 did not improve for more than "
                f"{args.early_stopping_patience} fine-tuning epochs."
            )
            break

    elapsed = time.time() - start_time
    history_path = output_dir / f"{run_stem}_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    print("Training complete.")
    print(f"Best checkpoint: {best_path}")
    print(f"Latest checkpoint: {latest_path}")
    print(f"History: {history_path}")
    print(f"Elapsed seconds: {elapsed:.1f}")


def print_epoch_summary(epoch, stage, train_metrics, val_metrics):
    print(
        f"Epoch {epoch:03d} [{stage}] "
        f"train_loss={train_metrics['loss']:.4f} "
        f"train_acc={train_metrics['accuracy']:.4f} "
        f"val_loss={val_metrics['loss']:.4f} "
        f"val_acc={val_metrics['accuracy']:.4f} "
        f"val_precision={val_metrics['precision']:.4f} "
        f"val_recall={val_metrics['recall']:.4f} "
        f"val_f1={val_metrics['f1']:.4f}"
    )


def add_common_training_args(parser, defaults):
    parser.add_argument("--data-dir", default="datasets/liveness")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--batch-size", type=int, default=defaults["batch_size"])
    parser.add_argument("--workers", type=int, default=defaults["workers"])
    parser.add_argument("--head-epochs", type=int, default=defaults["head_epochs"])
    parser.add_argument(
        "--finetune-epochs", type=int, default=defaults["finetune_epochs"]
    )
    parser.add_argument(
        "--unfreeze-blocks", type=int, default=defaults["unfreeze_blocks"]
    )
    parser.add_argument("--head-lr", type=float, default=defaults["head_lr"])
    parser.add_argument(
        "--finetune-lr", type=float, default=defaults["finetune_lr"]
    )
    parser.add_argument(
        "--weight-decay", type=float, default=defaults["weight_decay"]
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=defaults["early_stopping_patience"],
        help=(
            "Stop fine-tuning if val_f1 does not improve for more than this "
            "many epochs. Set to 0 to disable through patience."
        ),
    )
    parser.add_argument(
        "--disable-early-stopping",
        action="store_true",
        help="Always run all fine-tuning epochs even if validation F1 stops improving.",
    )
    parser.add_argument("--device", default=None, help="Example: cuda, cuda:0, or cpu")
    parser.add_argument(
        "--no-pretrained-backbone",
        action="store_true",
        help="Train from random initialization instead of ImageNet transfer learning.",
    )
