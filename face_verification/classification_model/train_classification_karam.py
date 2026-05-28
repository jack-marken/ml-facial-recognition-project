"""Train a supervised classification face recognition model (Karam).

Treats face recognition as a closed-set classification problem. The backbone
is fine-tuned with CrossEntropyLoss; after training, the classification head
is discarded and the backbone's 512-dim output is used as a face embedding.

Usage:
    python -m face_verification.classification_model.train_classification_karam \\
        --data-dir datasets/classification_data/train_data \\
        --val-dir  datasets/classification_data/val_data  \\
        --output   models/recognition_classification_karam.pth \\
        --epochs   30

Dataset layout (same as metric-learning module):
    data_dir/
        identity_A/   img1.jpg  img2.jpg  ...
        identity_B/   ...
"""
# Author: Karam

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .classification_model_karam import FaceClassificationModel


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

VAL_TRANSFORMS = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class FaceClassificationDataset(Dataset):
    """Loads face images from identity subfolders and assigns integer class labels."""

    def __init__(self, data_dir: str | Path, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        self.identity_to_label: dict[str, int] = {}
        self.label_to_identity: dict[int, str] = {}
        self._load_dataset()

    def _load_dataset(self) -> None:
        identity_dirs = sorted(p for p in self.data_dir.iterdir() if p.is_dir())
        if not identity_dirs:
            raise RuntimeError(f"No identity folders found in {self.data_dir}")

        for label, identity_dir in enumerate(identity_dirs):
            identity = identity_dir.name
            self.identity_to_label[identity] = label
            self.label_to_identity[label] = identity
            for image_path in sorted(identity_dir.iterdir()):
                if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((image_path, label))

        if not self.samples:
            raise RuntimeError(f"No images found under {self.data_dir}")

    @property
    def num_classes(self) -> int:
        return len(self.identity_to_label)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise ValueError(f"Could not read image: {image_path}")

        resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            return self.transform(rgb), label

        tensor = torch.from_numpy(rgb.astype("float32") / 255.0).permute(2, 0, 1)
        return tensor, label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train classification-based face recognition (Karam)."
    )
    parser.add_argument("--data-dir",       default="datasets/classification_data/train_data")
    parser.add_argument("--val-dir",        default="datasets/classification_data/val_data")
    parser.add_argument("--output",         default="models/recognition_classification_karam.pth")
    parser.add_argument("--epochs",         type=int,   default=30)
    parser.add_argument("--batch-size",     type=int,   default=32)
    parser.add_argument("--learning-rate",  type=float, default=1e-4)
    parser.add_argument("--num-workers",    type=int,   default=0)
    parser.add_argument(
        "--train-backbone", action="store_true",
        help="Unfreeze all backbone layers. Default: freeze all except layer4.",
    )
    return parser.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = FaceClassificationDataset(args.data_dir, transform=TRAIN_TRANSFORMS)
    val_dataset   = FaceClassificationDataset(args.val_dir,  transform=VAL_TRANSFORMS)

    num_classes = train_dataset.num_classes
    print(f"Identities : {num_classes}")
    print(f"Train images: {len(train_dataset)}   Val images: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    model = FaceClassificationModel(num_classes=num_classes, pretrained=True)
    if not args.train_backbone:
        _freeze_backbone_except_last_block(model)
    model.to(device)

    loss_fn   = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = _run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_loss,   val_acc   = _run_epoch(model, val_loader,   loss_fn, device, optimizer=None)
        scheduler.step()

        print(
            f"Epoch {epoch:02d}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "architecture":       "resnet34",
                    "num_classes":        num_classes,
                    "identity_to_label":  train_dataset.identity_to_label,
                    "label_to_identity":  train_dataset.label_to_identity,
                    "model_state_dict":   model.state_dict(),
                    "val_accuracy":       best_val_acc,
                },
                output_path,
            )
            print(f"  ↳ Saved best model (val_acc={best_val_acc:.3f}) → {output_path}")


def _run_epoch(
    model: FaceClassificationModel,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, float]:
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss    = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if optimizer is not None:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            logits = model(images)
            loss   = loss_fn(logits, labels)

        if optimizer is not None:
            loss.backward()
            optimizer.step()

        preds          = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_loss    += float(loss.item()) * images.size(0)
        total_samples += images.size(0)

    avg_loss = total_loss    / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)
    return avg_loss, accuracy


def _freeze_backbone_except_last_block(model: FaceClassificationModel) -> None:
    """Freeze all backbone layers except the final residual block (layer4)."""
    for param in model.backbone.parameters():
        param.requires_grad = False
    last_block = getattr(model.backbone, "layer4", None)
    if last_block is not None:
        for param in last_block.parameters():
            param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True


if __name__ == "__main__":
    main()
