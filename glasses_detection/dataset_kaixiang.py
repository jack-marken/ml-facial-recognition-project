import random
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_TO_LABEL = {
    "without_glasses": 0,
    "with_glasses": 1,
}
LABEL_TO_CLASS = {
    0: "without_glasses",
    1: "with_glasses",
}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class GlassesDataset(Dataset):
    def __init__(self, split_dir, train=False, max_per_class=None, seed=42):
        self.split_dir = Path(split_dir)
        self.train = train
        self.samples = load_glasses_samples(
            self.split_dir,
            max_per_class=max_per_class,
            seed=seed,
        )
        self.transform = make_glasses_transform(train=train)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), torch.tensor(float(label), dtype=torch.float32)


def load_glasses_samples(split_dir, max_per_class=None, seed=42):
    split_dir = Path(split_dir)
    if not split_dir.exists():
        raise FileNotFoundError(f"Glasses split directory not found: {split_dir}")

    samples = []
    for class_name, label in CLASS_TO_LABEL.items():
        class_dir = split_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing glasses class folder: {class_dir}")
        image_paths = [
            path
            for path in sorted(class_dir.iterdir())
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if max_per_class is not None and len(image_paths) > max_per_class:
            rng = random.Random(seed + label)
            image_paths = sorted(rng.sample(image_paths, max_per_class))
        samples.extend((image_path, label) for image_path in image_paths)

    if not samples:
        raise RuntimeError(f"No glasses images found under: {split_dir}")
    return samples


def make_glasses_transform(train=False):
    if train:
        return transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=8),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def face_image_to_tensor(face_image):
    image = Image.fromarray(face_image).convert("RGB")
    return make_glasses_transform(train=False)(image)


def count_split_images(data_dir):
    data_dir = Path(data_dir)
    counts = {}
    for split in ("train", "val", "test"):
        counts[split] = {}
        for class_name in CLASS_TO_LABEL:
            class_dir = data_dir / split / class_name
            counts[split][class_name] = (
                len(
                    [
                        path
                        for path in class_dir.iterdir()
                        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
                    ]
                )
                if class_dir.exists()
                else 0
            )
    return counts
