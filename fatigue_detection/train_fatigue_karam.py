"""Train an open/closed eye classifier for fatigue detection (Karam — Innovative Feature).

Model: EfficientNet-B0 fine-tuned as a binary classifier on 64×64 eye crops.
The trained model is the core of the PERCLOS-based fatigue detector.

Recommended dataset:
  MRL Eye Dataset  — http://mrl.cs.vsb.cz/eyedataset
  CEW Dataset      — https://parnec.nuaa.edu.cn/xtan/data/ClosedEyeDatabases.html

Expected folder layout after downloading and organising:
    datasets/eye_state/
        train/
            open/    (eye images, any resolution — will be resized to 64×64)
            closed/
        val/
            open/
            closed/

Usage:
    python -m fatigue_detection.train_fatigue_karam \\
        --data-dir datasets/eye_state/train \\
        --val-dir  datasets/eye_state/val   \\
        --output   models/fatigue_eye_karam.pth \\
        --epochs   25
"""
# Author: Karam (Innovative Feature — D/HD)

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
EYE_SIZE = (64, 64)   # compact input — eye crops are small and low-detail

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.4, contrast=0.4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

VAL_TRANSFORMS = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

CLASS_LABELS = ["closed", "open"]   # label 0 = closed, 1 = open


class EyeStateDataset(Dataset):
    """Binary dataset: open (1) vs closed (0) eye images."""

    FOLDER_TO_LABEL = {"closed": 0, "open": 1}

    def __init__(self, data_dir: str | Path, transform=None):
        self.data_dir  = Path(data_dir)
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        self._load_dataset()

    def _load_dataset(self) -> None:
        for cls_dir in sorted(p for p in self.data_dir.iterdir() if p.is_dir()):
            label = self.FOLDER_TO_LABEL.get(cls_dir.name.lower())
            if label is None:
                print(f"  Warning: unrecognised class folder '{cls_dir.name}' — skipping.")
                continue
            for image_path in sorted(cls_dir.iterdir()):
                if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((image_path, label))

        if not self.samples:
            raise RuntimeError(f"No eye images found under {self.data_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise ValueError(f"Could not read image: {image_path}")

        resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_CUBIC)
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            return self.transform(rgb), label

        tensor = torch.from_numpy(rgb.astype("float32") / 255.0).permute(2, 0, 1)
        return tensor, label


class EyeStateModel(nn.Module):
    """Lightweight EfficientNet-B0 binary classifier for eye state."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights  = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)
        in_feats = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        self.backbone   = backbone
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_feats, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),   # binary: closed / open
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(x))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train open/closed eye classifier for fatigue detection (Karam)."
    )
    parser.add_argument("--data-dir",      default="datasets/eye_state/train")
    parser.add_argument("--val-dir",       default="datasets/eye_state/val")
    parser.add_argument("--output",        default="models/fatigue_eye_karam.pth")
    parser.add_argument("--epochs",        type=int,   default=25)
    parser.add_argument("--batch-size",    type=int,   default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-workers",   type=int,   default=0)
    return parser.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = EyeStateDataset(args.data_dir, transform=TRAIN_TRANSFORMS)
    val_dataset   = EyeStateDataset(args.val_dir,  transform=VAL_TRANSFORMS)
    print(f"Train: {len(train_dataset)}  Val: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    model = EyeStateModel(pretrained=True)
    # Freeze backbone except last block
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.backbone.features[-1].parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True

    model.to(device)
    loss_fn   = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.learning_rate
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

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
                    "model_state_dict": model.state_dict(),
                    "val_accuracy":     best_val_acc,
                    "class_labels":     CLASS_LABELS,
                    "architecture":     "efficientnet_b0",
                },
                output_path,
            )
            print(f"  ↳ Saved best model (val_acc={best_val_acc:.3f}) → {output_path}")


def _run_epoch(model, loader, loss_fn, device, optimizer):
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss, total_correct, total_samples = 0.0, 0, 0

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

    return total_loss / max(total_samples, 1), total_correct / max(total_samples, 1)


if __name__ == "__main__":
    main()
