from pathlib import Path

from glasses_detection.dataset_kaixiang import CLASS_TO_LABEL, count_split_images


def main():
    data_dir = Path("datasets/glasses")
    counts = count_split_images(data_dir)
    print(f"Checking glasses dataset: {data_dir}")
    print("-" * 72)

    ok = True
    for split in ("train", "val", "test"):
        split_total = 0
        for class_name in CLASS_TO_LABEL:
            count = counts[split][class_name]
            split_total += count
            status = "OK" if count > 0 else "MISSING"
            if count <= 0:
                ok = False
            print(f"[{status:<7}] {split:<5}/{class_name:<15}: {count:6d} images")
        print(f"         {split:<5} total: {split_total:6d} images")
    print("-" * 72)

    train_positive = counts["train"]["with_glasses"]
    train_negative = counts["train"]["without_glasses"]
    if train_positive and train_negative:
        imbalance = max(train_positive, train_negative) / min(train_positive, train_negative)
        print(f"Train class imbalance ratio: {imbalance:.2f}:1")
        if imbalance > 1.25:
            print("Note: training should use positive class weighting or balanced sampling.")

    if not ok:
        raise SystemExit("Dataset check failed. Expected train/val/test with both glasses classes.")
    print("Dataset check passed. Structure is compatible with glasses detection training.")


if __name__ == "__main__":
    main()
