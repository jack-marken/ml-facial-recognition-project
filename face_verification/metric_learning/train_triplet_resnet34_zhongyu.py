"""Train ResNet34 face embeddings with triplet loss."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .embedding_model_zhongyu import (
    FaceEmbeddingModel,
    load_embedding_model,
    preprocess_face_image,
)
from .reporting_recognition_zhongyu import save_recognition_report


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


class TripletFaceDataset(Dataset):
    """Samples anchor, positive, and negative face images."""

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.identity_to_images = _load_identity_images(self.data_dir)
        self.identities = sorted(self.identity_to_images)
        self.anchor_items = [
            (identity, image_path)
            for identity, image_paths in self.identity_to_images.items()
            for image_path in image_paths
        ]

        if len(self.identities) < 2:
            raise ValueError("Triplet training needs at least two identities.")

        usable_identities = [
            identity
            for identity, image_paths in self.identity_to_images.items()
            if len(image_paths) >= 2
        ]
        if not usable_identities:
            raise ValueError("Triplet training needs at least one identity with 2+ images.")

    def __len__(self) -> int:
        return len(self.anchor_items)

    def __getitem__(self, index: int):
        anchor_identity, anchor_path = self.anchor_items[index]
        positive_candidates = [
            path
            for path in self.identity_to_images[anchor_identity]
            if path != anchor_path
        ]
        positive_path = random.choice(positive_candidates or self.identity_to_images[anchor_identity])

        negative_identity = random.choice(
            [identity for identity in self.identities if identity != anchor_identity]
        )
        negative_path = random.choice(self.identity_to_images[negative_identity])

        return (
            _load_face_tensor(anchor_path),
            _load_face_tensor(positive_path),
            _load_face_tensor(negative_path),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ResNet34 with triplet loss.")
    parser.add_argument("--data-dir", default="datasets/recognition/train")
    parser.add_argument("--val-dir", default="datasets/recognition/val")
    parser.add_argument("--test-dir", default="datasets/recognition/test")
    parser.add_argument("--output", default="models/recognition_triplet_resnet34_zhongyu.pth")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--train-backbone", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--report-dir",
        default="face_verification/metric_learning/report_recognition_zhongyu",
        help="Directory for training tables and figures.",
    )
    parser.add_argument(
        "--max-eval-pairs",
        type=int,
        default=20000,
        help="Maximum number of verification pairs used for report evaluation.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = TripletFaceDataset(args.data_dir)
    val_dataset = TripletFaceDataset(args.val_dir)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = FaceEmbeddingModel(architecture="resnet34", pretrained=True)
    if not args.train_backbone:
        _freeze_backbone_except_last_block(model)

    model.to(device)
    loss_function = nn.TripletMarginLoss(margin=args.margin, p=2)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise RuntimeError("No trainable parameters found.")

    optimizer = torch.optim.Adam(trainable_parameters, lr=args.learning_rate)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = _run_epoch(
            model=model,
            loader=train_loader,
            loss_function=loss_function,
            device=device,
            optimizer=optimizer,
        )
        val_loss = _run_epoch(
            model=model,
            loader=val_loader,
            loss_function=loss_function,
            device=device,
            optimizer=None,
        )

        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(float(train_loss), 6),
                "val_loss": round(float(val_loss), 6),
            }
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "architecture": "resnet34",
                    "model_state_dict": model.state_dict(),
                    "val_loss": best_val_loss,
                    "margin": args.margin,
                },
                output_path,
            )
            print(f"Saved best model to {output_path}")

    best_model = load_embedding_model(
        architecture="resnet34",
        model_path=output_path,
        device=device,
        pretrained=False,
    )
    save_recognition_report(
        model=best_model,
        history=history,
        args=args,
        architecture="resnet34",
        output_path=output_path,
        report_root=Path(args.report_dir),
        embedding_size=model.embedding_dim,
    )
    print(f"Saved recognition report artifacts to {Path(args.report_dir) / 'resnet34'}")


def _run_epoch(
    model: FaceEmbeddingModel,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    if optimizer is None:
        model.eval()
    else:
        model.train()

    total_loss = 0.0
    total_samples = 0

    for anchor, positive, negative in loader:
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        if optimizer is not None:
            optimizer.zero_grad()

        with torch.set_grad_enabled(optimizer is not None):
            anchor_embedding = model(anchor)
            positive_embedding = model(positive)
            negative_embedding = model(negative)
            loss = loss_function(anchor_embedding, positive_embedding, negative_embedding)

        if optimizer is not None:
            loss.backward()
            optimizer.step()

        batch_size = anchor.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


def _freeze_backbone_except_last_block(model: FaceEmbeddingModel) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False

    # ResNet children: conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool, fc
    last_block = getattr(model.feature_extractor, "layer4", None)
    if last_block is None:
        for parameter in model.parameters():
            parameter.requires_grad = True
        return

    for parameter in last_block.parameters():
        parameter.requires_grad = True


def _load_identity_images(data_dir: Path) -> dict[str, list[Path]]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Recognition training directory not found: {data_dir}")

    identity_to_images: dict[str, list[Path]] = {}
    for identity_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        image_paths = [
            path
            for path in sorted(identity_dir.iterdir())
            if path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if image_paths:
            identity_to_images[identity_dir.name] = image_paths

    if not identity_to_images:
        raise RuntimeError(f"No recognition images found under {data_dir}.")

    return identity_to_images


def _load_face_tensor(image_path: Path) -> torch.Tensor:
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise ValueError(f"Failed to read image: {image_path}")

    resized_bgr = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    face_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
    return preprocess_face_image(face_rgb, device=torch.device("cpu")).squeeze(0)


if __name__ == "__main__":
    main()
