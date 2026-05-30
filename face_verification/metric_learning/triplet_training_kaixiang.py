import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

from face_verification.metric_learning.siamese_dataset_kaixiang import (
    load_face_tensor,
    load_identity_images,
)
from face_verification.metric_learning.siamese_models_kaixiang import pairwise_distance
from face_verification.metric_learning.siamese_training_kaixiang import (
    FixedPairDataset,
    calculate_pair_metrics,
    save_checkpoint,
)


class IdentityImageDataset(Dataset):
    def __init__(self, data_dir, augment=False):
        self.identity_to_images = load_identity_images(data_dir)
        self.identities = sorted(self.identity_to_images)
        self.identity_to_label = {
            identity: index for index, identity in enumerate(self.identities)
        }
        self.items = [
            (identity, image_path)
            for identity, image_paths in self.identity_to_images.items()
            for image_path in image_paths
        ]
        self.label_to_indices = {}
        for index, (identity, _) in enumerate(self.items):
            label = self.identity_to_label[identity]
            self.label_to_indices.setdefault(label, []).append(index)
        self.labels = sorted(self.label_to_indices)
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        identity, image_path = self.items[index]
        label = self.identity_to_label[identity]
        return load_face_tensor(image_path, augment=self.augment), torch.tensor(label)


class PKBatchSampler(Sampler):
    def __init__(self, label_to_indices, identities_per_batch, samples_per_identity, batches_per_epoch, seed=42):
        self.label_to_indices = label_to_indices
        self.labels = sorted(label_to_indices)
        self.identities_per_batch = identities_per_batch
        self.samples_per_identity = samples_per_identity
        self.batches_per_epoch = batches_per_epoch
        self.random = random.Random(seed)

        if len(self.labels) < identities_per_batch:
            raise ValueError("Not enough identities for PK batch sampling.")
        for label, indices in self.label_to_indices.items():
            if len(indices) < 2:
                raise ValueError(f"Identity label {label} needs at least 2 images for triplet training.")

    def __len__(self):
        return self.batches_per_epoch

    def __iter__(self):
        for _ in range(self.batches_per_epoch):
            batch = []
            selected_labels = self.random.sample(self.labels, self.identities_per_batch)
            for label in selected_labels:
                indices = self.label_to_indices[label]
                if len(indices) >= self.samples_per_identity:
                    batch.extend(self.random.sample(indices, self.samples_per_identity))
                else:
                    batch.extend(self.random.choices(indices, k=self.samples_per_identity))
            yield batch


class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        distance_matrix = torch.cdist(embeddings, embeddings, p=2)
        labels = labels.view(-1, 1)
        positive_mask = labels.eq(labels.t())
        negative_mask = ~positive_mask
        positive_mask.fill_diagonal_(False)

        hardest_positive = distance_matrix.masked_fill(~positive_mask, 0.0).max(dim=1).values
        hardest_negative = distance_matrix.masked_fill(~negative_mask, float("inf")).min(dim=1).values
        losses = torch.relu(hardest_positive - hardest_negative + self.margin)
        valid = torch.isfinite(hardest_negative)
        return losses[valid].mean()


def run_triplet_training(
    model_name,
    args,
    build_model,
    freeze_backbone,
    unfreeze_backbone,
):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    run_stem = f"recognition_triplet_{model_name}_kaixiang"
    if args.run_name:
        run_stem += f"_{args.run_name}"
    best_path = output_dir / f"{run_stem}_best.pth"
    latest_path = output_dir / f"{run_stem}_latest.pth"
    history_path = output_dir / f"{run_stem}_history.json"

    print(f"Training triplet model: {model_name}")
    print(f"Device: {device}")
    print(f"Dataset: {args.data_dir}")
    print(
        "Hyperparameters: "
        f"batch_size={args.identities_per_batch * args.samples_per_identity}, "
        f"P={args.identities_per_batch}, K={args.samples_per_identity}, "
        f"head_epochs={args.head_epochs}, finetune_epochs={args.finetune_epochs}, "
        f"margin={args.margin}, head_lr={args.head_lr}, finetune_lr={args.finetune_lr}"
    )

    train_dataset = IdentityImageDataset(Path(args.data_dir) / "train", augment=True)
    sampler = PKBatchSampler(
        train_dataset.label_to_indices,
        identities_per_batch=args.identities_per_batch,
        samples_per_identity=args.samples_per_identity,
        batches_per_epoch=args.batches_per_epoch,
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = make_pair_loader(args, split="val")

    model = build_model(pretrained_backbone=True)
    model.to(device)
    loss_function = BatchHardTripletLoss(margin=args.margin)
    history = []
    best_score = -1.0
    epoch_index = 0
    start_time = time.time()
    args_dict = vars(args).copy()

    freeze_backbone(model)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.head_lr,
        weight_decay=args.weight_decay,
    )

    for epoch in range(1, args.head_epochs + 1):
        epoch_index += 1
        train_metrics = run_triplet_epoch(model, train_loader, loss_function, device, optimizer)
        val_metrics = evaluate_pair_loader(model, val_loader, device)
        history.append({"epoch": epoch_index, "stage": "head", "train": train_metrics, "val": val_metrics})
        print_summary(epoch_index, "head", train_metrics, val_metrics)
        save_checkpoint(latest_path, model, model_name, epoch_index, val_metrics, args_dict)
        best_score = maybe_save_best(best_path, model, model_name, epoch_index, val_metrics, args_dict, best_score)

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
        previous_best = best_score
        train_metrics = run_triplet_epoch(model, train_loader, loss_function, device, optimizer)
        val_metrics = evaluate_pair_loader(model, val_loader, device)
        history.append({"epoch": epoch_index, "stage": "fine_tune", "train": train_metrics, "val": val_metrics})
        print_summary(epoch_index, "fine_tune", train_metrics, val_metrics)
        save_checkpoint(latest_path, model, model_name, epoch_index, val_metrics, args_dict)
        best_score = maybe_save_best(best_path, model, model_name, epoch_index, val_metrics, args_dict, best_score)

        if best_score <= previous_best:
            epochs_without_improvement += 1
        else:
            epochs_without_improvement = 0
        if epochs_without_improvement > args.early_stopping_patience:
            print("Early stopping triggered: validation AUC did not improve.")
            break

    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print("Triplet training complete.")
    print(f"Best checkpoint: {best_path}")
    print(f"Latest checkpoint: {latest_path}")
    print(f"History: {history_path}")
    print(f"Elapsed seconds: {time.time() - start_time:.1f}")


def make_pair_loader(args, split):
    dataset = FixedPairDataset(
        Path(args.data_dir) / split,
        max_positive_pairs_per_identity=args.max_positive_pairs_per_identity,
        max_negative_pairs=args.max_negative_pairs,
        seed=args.seed,
    )
    return DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )


def run_triplet_epoch(model, loader, loss_function, device, optimizer):
    model.train()
    total_loss = 0.0
    total_samples = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        embeddings = model.forward_once(images)
        loss = loss_function(embeddings, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * images.size(0)
        total_samples += images.size(0)
    return {"loss": total_loss / max(total_samples, 1)}


@torch.no_grad()
def evaluate_pair_loader(model, loader, device):
    model.eval()
    labels = []
    distances = []
    total_samples = 0
    start_time = time.perf_counter()
    for first_images, second_images, targets in loader:
        first_images = first_images.to(device)
        second_images = second_images.to(device)
        first_embeddings, second_embeddings = model(first_images, second_images)
        batch_distances = pairwise_distance(first_embeddings, second_embeddings)
        total_samples += first_images.size(0)
        labels.extend(targets.tolist())
        distances.extend(batch_distances.cpu().tolist())
    metrics = calculate_pair_metrics(labels, distances)
    metrics["fps"] = total_samples / max(time.perf_counter() - start_time, 1e-12)
    return metrics


def maybe_save_best(path, model, model_name, epoch, val_metrics, args_dict, best_score):
    score = val_metrics["roc_auc"] if val_metrics["roc_auc"] is not None else val_metrics["accuracy"]
    if score > best_score:
        save_checkpoint(path, model, model_name, epoch, val_metrics, args_dict)
        return score
    return best_score


def print_summary(epoch, stage, train_metrics, val_metrics):
    print(
        f"Epoch {epoch:03d} [{stage}] "
        f"train_loss={train_metrics['loss']:.4f} "
        f"val_acc={val_metrics['accuracy']:.4f} "
        f"val_auc={val_metrics['roc_auc']} "
        f"val_threshold={val_metrics['best_distance_threshold']:.4f}"
    )


def add_triplet_args(parser, defaults):
    parser.add_argument("--data-dir", default="datasets/recognition")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--head-epochs", type=int, default=defaults["head_epochs"])
    parser.add_argument("--finetune-epochs", type=int, default=defaults["finetune_epochs"])
    parser.add_argument("--unfreeze-blocks", type=int, default=defaults["unfreeze_blocks"])
    parser.add_argument("--identities-per-batch", type=int, default=defaults["identities_per_batch"])
    parser.add_argument("--samples-per-identity", type=int, default=defaults["samples_per_identity"])
    parser.add_argument("--batches-per-epoch", type=int, default=defaults["batches_per_epoch"])
    parser.add_argument("--eval-batch-size", type=int, default=defaults["eval_batch_size"])
    parser.add_argument("--head-lr", type=float, default=defaults["head_lr"])
    parser.add_argument("--finetune-lr", type=float, default=defaults["finetune_lr"])
    parser.add_argument("--weight-decay", type=float, default=defaults["weight_decay"])
    parser.add_argument("--margin", type=float, default=defaults["margin"])
    parser.add_argument("--max-positive-pairs-per-identity", type=int, default=defaults["max_positive_pairs_per_identity"])
    parser.add_argument("--max-negative-pairs", type=int, default=defaults["max_negative_pairs"])
    parser.add_argument("--early-stopping-patience", type=int, default=defaults["early_stopping_patience"])
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
