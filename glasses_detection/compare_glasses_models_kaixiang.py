import argparse
import json
from pathlib import Path

from glasses_detection.evaluate_glasses_kaixiang import evaluate_checkpoint


DEFAULT_MODELS = [
    {
        "name": "glasses_mobilenetv2",
        "checkpoint": "models/glasses_mobilenetv2_kaixiang_final_best.pth",
    },
    {
        "name": "glasses_efficientnetb0",
        "checkpoint": "models/glasses_efficientnetb0_kaixiang_final_best.pth",
    },
]


def main():
    parser = argparse.ArgumentParser(description="Compare Kaixiang glasses detection models.")
    parser.add_argument("--data-dir", default="datasets/glasses")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-json", type=Path, default=Path("reports/glasses_model_comparison_kaixiang.json"))
    args = parser.parse_args()

    results = []
    for model_config in DEFAULT_MODELS:
        checkpoint_path = Path(model_config["checkpoint"])
        print(f"Evaluating {model_config['name']}...")
        if not checkpoint_path.exists():
            results.append(
                {
                    "name": model_config["name"],
                    "checkpoint": str(checkpoint_path),
                    "error": "checkpoint not found",
                }
            )
            continue
        metrics = evaluate_checkpoint(
            checkpoint_path=checkpoint_path,
            data_dir=args.data_dir,
            split=args.split,
            batch_size=args.batch_size,
            workers=args.workers,
            device=args.device,
        )
        metrics["name"] = model_config["name"]
        results.append(metrics)

    comparison = {
        "split": args.split,
        "results": results,
    }
    print(json.dumps(comparison, indent=2))
    print("\nSummary:")
    print("model,accuracy,precision,recall,f1,fps")
    for result in results:
        if "error" in result:
            print(f"{result['name']},ERROR,{result['error']}")
            continue
        print(
            f"{result['name']},"
            f"{result['accuracy']},"
            f"{result['precision']},"
            f"{result['recall']},"
            f"{result['f1']},"
            f"{result['fps']}"
        )

    if args.save_json:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
        print(f"Saved comparison to: {args.save_json}")


if __name__ == "__main__":
    main()
