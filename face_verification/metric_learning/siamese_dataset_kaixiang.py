import random
from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_identity_images(data_dir: str | Path) -> dict[str, list[Path]]:
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Recognition dataset folder not found: {data_dir}")

    identity_to_images: dict[str, list[Path]] = {}
    for identity_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        image_paths = [
            path
            for path in sorted(identity_dir.rglob("*"))
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if image_paths:
            identity_to_images[identity_dir.name] = image_paths

    if len(identity_to_images) < 2:
        raise ValueError(f"Need at least two identities under {data_dir}")

    return identity_to_images


class SiamesePairDataset(Dataset):
    """Balanced online pair sampler for Siamese metric learning."""

    def __init__(self, data_dir, pairs_per_epoch=None, seed=42):
        self.identity_to_images = load_identity_images(data_dir)
        self.identities = sorted(self.identity_to_images)
        self.anchor_items = [
            (identity, image_path)
            for identity, image_paths in self.identity_to_images.items()
            for image_path in image_paths
        ]
        self.positive_identities = [
            identity
            for identity, image_paths in self.identity_to_images.items()
            if len(image_paths) >= 2
        ]
        if not self.positive_identities:
            raise ValueError("At least one identity must have 2+ images.")

        self.pairs_per_epoch = pairs_per_epoch or max(2 * len(self.anchor_items), 1000)
        self.random = random.Random(seed)

    def __len__(self):
        return self.pairs_per_epoch

    def __getitem__(self, index):
        make_positive_pair = index % 2 == 0

        if make_positive_pair:
            identity = self.random.choice(self.positive_identities)
            first_path, second_path = self.random.sample(
                self.identity_to_images[identity],
                2,
            )
            label = 1.0
        else:
            first_identity, second_identity = self.random.sample(self.identities, 2)
            first_path = self.random.choice(self.identity_to_images[first_identity])
            second_path = self.random.choice(self.identity_to_images[second_identity])
            label = 0.0

        return (
            load_face_tensor(first_path),
            load_face_tensor(second_path),
            torch.tensor(label, dtype=torch.float32),
        )


class FixedPairDataset(Dataset):
    """Deterministic pair dataset for validation and test evaluation."""

    def __init__(
        self,
        data_dir,
        max_positive_pairs_per_identity=20,
        max_negative_pairs=1000,
        seed=42,
    ):
        identity_to_images = load_identity_images(data_dir)
        self.pairs = build_fixed_pairs(
            identity_to_images,
            max_positive_pairs_per_identity=max_positive_pairs_per_identity,
            max_negative_pairs=max_negative_pairs,
            seed=seed,
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        first_path, second_path, label = self.pairs[index]
        return (
            load_face_tensor(first_path),
            load_face_tensor(second_path),
            torch.tensor(label, dtype=torch.float32),
        )


def build_fixed_pairs(
    identity_to_images: dict[str, list[Path]],
    max_positive_pairs_per_identity: int,
    max_negative_pairs: int,
    seed: int,
) -> list[tuple[Path, Path, float]]:
    random_generator = random.Random(seed)
    pairs: list[tuple[Path, Path, float]] = []

    for identity, image_paths in sorted(identity_to_images.items()):
        positive_pairs = []
        for first_index in range(len(image_paths)):
            for second_index in range(first_index + 1, len(image_paths)):
                positive_pairs.append((image_paths[first_index], image_paths[second_index], 1.0))

        random_generator.shuffle(positive_pairs)
        pairs.extend(positive_pairs[:max_positive_pairs_per_identity])

    identities = sorted(identity_to_images)
    negative_pairs = []
    for first_index in range(len(identities)):
        for second_index in range(first_index + 1, len(identities)):
            first_identity = identities[first_index]
            second_identity = identities[second_index]
            for first_path in identity_to_images[first_identity]:
                for second_path in identity_to_images[second_identity]:
                    negative_pairs.append((first_path, second_path, 0.0))

    random_generator.shuffle(negative_pairs)
    positive_count = sum(1 for _, _, label in pairs if label == 1.0)
    negative_limit = min(max_negative_pairs, max(positive_count, 1), len(negative_pairs))
    pairs.extend(negative_pairs[:negative_limit])

    random_generator.shuffle(pairs)
    if not pairs:
        raise RuntimeError("No evaluation pairs could be generated.")
    return pairs


def load_face_tensor(image_path: Path) -> torch.Tensor:
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise ValueError(f"Failed to read image: {image_path}")

    resized_bgr = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
    return face_image_to_tensor(rgb)


def face_image_to_tensor(face_image) -> torch.Tensor:
    tensor = torch.from_numpy(face_image.astype("float32") / 255.0).permute(2, 0, 1)
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD
