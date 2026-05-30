from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXPECTED_SPLITS = ("train", "val", "test")


def count_images(folder: Path) -> int:
    return sum(
        1
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def check_metric_dataset(dataset_root: Path) -> bool:
    print(f"Checking metric-learning dataset: {dataset_root}")
    print("-" * 80)

    ok = True
    if not dataset_root.exists():
        print(f"[ERROR] Dataset folder does not exist: {dataset_root}")
        return False

    for split in EXPECTED_SPLITS:
        split_dir = dataset_root / split
        if not split_dir.exists():
            print(f"[MISSING] {split}: {split_dir}")
            ok = False
            continue

        identity_dirs = sorted(path for path in split_dir.iterdir() if path.is_dir())
        split_total = 0
        usable_for_positive_pairs = 0

        print(f"[SPLIT] {split}")
        for identity_dir in identity_dirs:
            image_count = count_images(identity_dir)
            split_total += image_count
            if image_count >= 2:
                usable_for_positive_pairs += 1
            status = "OK" if image_count >= 2 else "LOW"
            print(f"  [{status:3}] {identity_dir.name:25} {image_count:5} images")

        if len(identity_dirs) < 2:
            print(f"  [ERROR] {split} needs at least 2 identities.")
            ok = False

        if usable_for_positive_pairs == 0:
            print(f"  [ERROR] {split} needs at least one identity with 2+ images.")
            ok = False

        print(f"  identities={len(identity_dirs)} total_images={split_total}")
        print("-" * 80)

    if ok:
        print("Dataset check passed. Structure is compatible with Siamese training.")
    else:
        print("Dataset check failed. Fix missing/low-count folders before training.")

    return ok


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Check Kaixiang metric-learning recognition dataset."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("datasets/recognition"),
        help="Path containing train/val/test identity folders.",
    )
    args = parser.parse_args()

    raise SystemExit(0 if check_metric_dataset(args.data_dir) else 1)
