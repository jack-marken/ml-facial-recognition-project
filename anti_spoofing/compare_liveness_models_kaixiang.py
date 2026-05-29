import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from anti_spoofing.liveness_dataset_kaixiang import IMAGE_EXTENSIONS, LABEL_TO_INDEX
from anti_spoofing.liveness_training_kaixiang import calculate_metrics


DEFAULT_MODELS = [
    {
        "name": "kaixiang_mobilenetv2",
        "owner": "kaixiang",
        "framework": "torch",
        "checkpoint": Path("models/liveness_mobilenetv2_kaixiang_final1_best.pth"),
        "threshold": 0.5,
    },
    {
        "name": "kaixiang_efficientnetb0",
        "owner": "kaixiang",
        "framework": "torch",
        "checkpoint": Path("models/liveness_efficientnetb0_kaixiang_final1_best.pth"),
        "threshold": 0.5,
    },
    {
        "name": "zhongyu_densenet121",
        "owner": "zhongyu",
        "framework": "keras",
        "checkpoint": Path("models/liveness_densenet121_zhongyu.keras"),
        "threshold": 0.5,
    },
    {
        "name": "zhongyu_resnet50v2",
        "owner": "zhongyu",
        "framework": "keras",
        "checkpoint": Path("models/liveness_resnet50v2_zhongyu.weights.h5"),
        "threshold": 0.5,
    },
]


def collect_samples(data_dir, split):
    split_dir = Path(data_dir) / split
    samples = []
    for class_name, label in LABEL_TO_INDEX.items():
        class_dir = split_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing dataset folder: {class_dir}")

        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((path, label))

    if not samples:
        raise RuntimeError(f"No liveness images found in {split_dir}")
    return samples


def try_roc_auc(labels, probabilities):
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(labels, probabilities))
    except Exception:
        return None


def metrics_at_best_f1(labels, probabilities):
    thresholds = sorted(set(float(prob) for prob in probabilities))
    thresholds = [0.0] + thresholds + [1.0]
    best = None

    for threshold in thresholds:
        metrics = calculate_metrics(labels, probabilities, threshold=threshold)
        metrics["threshold"] = threshold
        if best is None or metrics["f1"] > best["f1"]:
            best = metrics

    return best


def format_result(model_config, labels, probabilities, elapsed):
    threshold = model_config["threshold"]
    metrics = calculate_metrics(labels, probabilities, threshold=threshold)
    metrics["roc_auc"] = try_roc_auc(labels, probabilities)
    metrics["threshold"] = threshold
    metrics["best_f1_threshold"] = metrics_at_best_f1(labels, probabilities)
    metrics["fps"] = len(labels) / max(1e-12, elapsed)
    metrics["samples"] = len(labels)
    metrics["name"] = model_config["name"]
    metrics["owner"] = model_config["owner"]
    metrics["framework"] = model_config["framework"]
    metrics["checkpoint"] = str(model_config["checkpoint"])
    metrics["confusion_matrix"] = {
        "rows": ["true_real", "true_spoof"],
        "cols": ["pred_real", "pred_spoof"],
        "values": [
            [metrics["tp"], metrics["fn"]],
            [metrics["fp"], metrics["tn"]],
        ],
    }
    return metrics


def evaluate_torch_model(model_config, data_dir, split, batch_size, workers, device_arg):
    import torch
    from torch.utils.data import DataLoader

    from anti_spoofing.liveness_dataset_kaixiang import LivenessImageDataset
    from anti_spoofing.liveness_models_kaixiang import load_checkpoint
    from anti_spoofing.liveness_training_kaixiang import make_transforms

    device = torch.device(device_arg or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, _, device = load_checkpoint(model_config["checkpoint"], device=device)
    dataset = LivenessImageDataset(
        data_dir,
        split=split,
        transform=make_transforms(train=False),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )

    labels = []
    probabilities = []

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            logits = model(images).squeeze(1)
            probs = torch.sigmoid(logits).detach().cpu().tolist()
            probabilities.extend(probs)
            labels.extend(targets.int().tolist())

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return format_result(model_config, labels, probabilities, elapsed)


def load_keras_liveness_model(checkpoint_path):
    try:
        from anti_spoofing.liveness_zhongyu import _load_model
    except ImportError as error:
        raise ImportError(
            "Zhongyu Keras liveness files are missing. Restore them from "
            "origin/feature/retraining-evaluation-zhongyu first."
        ) from error

    return _load_model(Path(checkpoint_path))


def make_keras_batch(paths):
    images = []
    for path in paths:
        image = Image.open(path).convert("RGB").resize((224, 224))
        images.append(np.asarray(image, dtype=np.float32))
    return np.stack(images, axis=0)


def evaluate_keras_model(model_config, samples, batch_size):
    try:
        import tensorflow as tf
    except ImportError as error:
        raise ImportError(
            "TensorFlow is required to evaluate Zhongyu's Keras liveness models. "
            "Install it in the current environment before running this comparison."
        ) from error

    tf.get_logger().setLevel("ERROR")
    model = load_keras_liveness_model(model_config["checkpoint"])

    labels = []
    probabilities = []

    start = time.perf_counter()
    for start_index in range(0, len(samples), batch_size):
        batch_samples = samples[start_index : start_index + batch_size]
        paths = [path for path, _ in batch_samples]
        batch_labels = [label for _, label in batch_samples]

        batch = make_keras_batch(paths)
        raw_probs = model.predict(batch, verbose=0).reshape(-1)
        probabilities.extend(np.clip(raw_probs, 0.0, 1.0).astype(float).tolist())
        labels.extend(batch_labels)

    elapsed = time.perf_counter() - start
    return format_result(model_config, labels, probabilities, elapsed)


def evaluate_model(model_config, data_dir, split, batch_size, workers, device_arg, samples):
    checkpoint = model_config["checkpoint"]
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint for {model_config['name']}: {checkpoint}")

    if model_config["framework"] == "torch":
        return evaluate_torch_model(
            model_config,
            data_dir=data_dir,
            split=split,
            batch_size=batch_size,
            workers=workers,
            device_arg=device_arg,
        )

    if model_config["framework"] == "keras":
        return evaluate_keras_model(
            model_config,
            samples=samples,
            batch_size=batch_size,
        )

    raise ValueError(f"Unsupported framework: {model_config['framework']}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare Kaixiang and Zhongyu liveness models on the same split."
    )
    parser.add_argument("--data-dir", default="datasets/liveness")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--save-json",
        type=Path,
        default=Path("reports/liveness_model_comparison_kaixiang.json"),
    )
    args = parser.parse_args()

    samples = collect_samples(args.data_dir, args.split)
    results = []

    print(f"Dataset: {Path(args.data_dir) / args.split}")
    print(f"Samples: {len(samples)}")
    for model_config in DEFAULT_MODELS:
        print(f"Evaluating {model_config['name']}...")
        result = evaluate_model(
            model_config,
            data_dir=args.data_dir,
            split=args.split,
            batch_size=args.batch_size,
            workers=args.workers,
            device_arg=args.device,
            samples=samples,
        )
        results.append(result)

    comparison = {
        "split": args.split,
        "samples": len(samples),
        "results": results,
    }

    print(json.dumps(comparison, indent=2))
    print("\nSummary:")
    print("model,accuracy,precision,recall,f1,roc_auc,threshold,fps")
    for result in results:
        print(
            f"{result['name']},"
            f"{result['accuracy']},"
            f"{result['precision']},"
            f"{result['recall']},"
            f"{result['f1']},"
            f"{result['roc_auc']},"
            f"{result['threshold']},"
            f"{result['fps']}"
        )

    if args.save_json is not None:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
        print(f"Saved comparison to: {args.save_json}")


if __name__ == "__main__":
    main()
