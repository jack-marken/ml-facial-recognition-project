"""Train the emotion detection model on FER-2013 (Karam).

Dataset: FER-2013 from Kaggle (https://www.kaggle.com/datasets/msambare/fer2013)
Download and extract so the folder layout is:
    datasets/fer2013/
        train/
            angry/   disgust/  fear/  happy/  neutral/  sad/  surprise/
        val/
            angry/   ...

Usage:
    python -m emotion_detection.train_emotion_karam \\
        --data-dir datasets/fer2013/train \\
        --val-dir  datasets/fer2013/val   \\
        --output   models/emotion_karam.pth \\
        --epochs   40
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

from .emotion_model_karam import EMOTION_LABELS, EmotionModel


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# Augmentation is important for FER-2013 which is small and noisy
TRAIN_TRANSFORMS = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

VAL_TRANSFORMS = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class FER2013Dataset(Dataset):
    """Loads FER-2013 images from class subfolders.

    Expects the standard Kaggle folder layout with one subfolder per emotion.
    Images are resized from 48×48 to 224×224 to match EfficientNet input.
    """

    # Map folder names → canonical label indices matching EMOTION_LABELS
    FOLDER_TO_LABEL: dict[str, int] = {
        "angry":    0,
        "disgust":  1,
        "fear":     2,
        "happy":    3,
        "neutral":  4,
        "sad":      5,
        "surprise": 6,
    }

    def __init__(self, data_dir: str | Path, transform=None):
        self.data_dir  = Path(data_dir)
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        self._load_dataset()

    def _load_dataset(self) -> None:
        for emotion_dir in sorted(p for p in self.data_dir.iterdir() if p.is_dir()):
            label = self.FOLDER_TO_LABEL.get(emotion_dir.name.lower())
            if label is None:
                print(f"  Warning: unrecognised emotion folder '{emotion_dir.name}' — skipping.")
                continue
            for image_path in sorted(emotion_dir.iterdir()):
                if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((image_path, label))

        if not self.samples:
            raise RuntimeError(f"No images found under {self.data_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise ValueError(f"Could not read image: {image_path}")

        # FER-2013 images are 48×48; resize to 224×224 for EfficientNet
        resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_CUBIC)
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            return self.transform(rgb), label

        tensor = torch.from_numpy(rgb.astype("float32") / 255.0).permute(2, 0, 1)
        return tensor, label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train emotion detection model on FER-2013 (Karam)."
    )
    parser.add_argument("--data-dir",      default="datasets/fer2013/train")
    parser.add_argument("--val-dir",       default="datasets/fer2013/val")
    parser.add_argument("--output",        default="models/emotion_karam.pth")
    parser.add_argument("--epochs",        type=int,   default=40)
    parser.add_argument("--batch-size",    type=int,   default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-workers",   type=int,   default=0)
    parser.add_argument(
        "--train-backbone", action="store_true",
        help="Unfreeze full EfficientNet backbone (default: freeze all but last block).",
    )
    return parser.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = FER2013Dataset(args.data_dir, transform=TRAIN_TRANSFORMS)
    val_dataset   = FER2013Dataset(args.val_dir,  transform=VAL_TRANSFORMS)
    print(f"Train: {len(train_dataset)} images   Val: {len(val_dataset)} images")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    model = EmotionModel(pretrained=True)
    if not args.train_backbone:
        _freeze_backbone_except_last_block(model)
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
                    "num_classes":      7,
                    "emotion_labels":   EMOTION_LABELS,
                    "model_state_dict": model.state_dict(),
                    "val_accuracy":     best_val_acc,
                    "architecture":     "efficientnet_b0",
                },
                output_path,
            )
            print(f"  ↳ Saved best model (val_acc={best_val_acc:.3f}) → {output_path}")


def _run_epoch(
    model:     EmotionModel,
    loader:    DataLoader,
    loss_fn:   nn.Module,
    device:    torch.device,
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

    return total_loss / max(total_samples, 1), total_correct / max(total_samples, 1)


def _freeze_backbone_except_last_block(model: EmotionModel) -> None:
    """Freeze all EfficientNet features except the last MBConv block."""
    for param in model.backbone.parameters():
        param.requires_grad = False

    # EfficientNet features[-1] is the last convolutional block
    last_block = model.backbone.features[-1]
    for param in last_block.parameters():
        param.requires_grad = True

    for param in model.classifier.parameters():
        param.requires_grad = True


if __name__ == "__main__":
    main()
