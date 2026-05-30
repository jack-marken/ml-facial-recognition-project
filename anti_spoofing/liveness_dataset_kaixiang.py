from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABEL_TO_INDEX = {"spoof": 0, "real": 1}
INDEX_TO_LABEL = {0: "SPOOF", 1: "REAL"}


class LivenessImageDataset(Dataset):
    """Binary liveness dataset with an explicit REAL=1, SPOOF=0 label map."""

    def __init__(self, root_dir, split, transform=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.samples = []

        split_dir = self.root_dir / split
        for class_name, label in LABEL_TO_INDEX.items():
            class_dir = split_dir / class_name
            if not class_dir.exists():
                raise FileNotFoundError(f"Missing dataset folder: {class_dir}")

            for path in sorted(class_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((path, label))

        if not self.samples:
            raise RuntimeError(f"No images found in {split_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label
