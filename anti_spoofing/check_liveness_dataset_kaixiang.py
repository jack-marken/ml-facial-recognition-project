from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXPECTED_SPLITS = ("train", "val", "test")
EXPECTED_CLASSES = ("real", "spoof")


def count_images(folder: Path) -> int:
    return sum(
        1
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def check_liveness_dataset(dataset_root: Path) -> bool:
    print(f"Checking liveness dataset: {dataset_root}")
    print("-" * 72)

    ok = True
    split_totals = {}

    if not dataset_root.exists():
        print(f"[ERROR] Dataset folder does not exist: {dataset_root}")
        return False

    for split in EXPECTED_SPLITS:
        split_total = 0
        for class_name in EXPECTED_CLASSES:
            class_dir = dataset_root / split / class_name
            if not class_dir.exists():
                print(f"[MISSING] {split}/{class_name}: {class_dir}")
                ok = False
                continue

            image_count = count_images(class_dir)
            split_total += image_count
            status = "OK" if image_count > 0 else "EMPTY"
            print(f"[{status:7}] {split:5}/{class_name:5}: {image_count:5} images")

            if image_count == 0:
                ok = False

        split_totals[split] = split_total

    print("-" * 72)
    print("Split totals:")
    for split, total in split_totals.items():
        print(f"  {split:5}: {total:5} images")

    print("-" * 72)
    if ok:
        print("Dataset check passed. Structure is compatible with liveness training.")
    else:
        print("Dataset check failed. Fix missing or empty folders before training.")

    return ok


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Check Kaixiang liveness dataset folder structure."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("datasets/liveness"),
        help="Path to datasets/liveness.",
    )
    args = parser.parse_args()

    raise SystemExit(0 if check_liveness_dataset(args.data_dir) else 1)
