import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description="Plot final metric-recognition model comparison from comparison JSON."
    )
    parser.add_argument(
        "--comparison-json",
        type=Path,
        default=Path("reports/metric_model_comparison_kaixiang.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    if not args.comparison_json.exists():
        raise FileNotFoundError(
            f"Comparison JSON not found: {args.comparison_json}. "
            "Run compare_metric_models_kaixiang.py first."
        )

    with args.comparison_json.open("r", encoding="utf-8") as file:
        comparison = json.load(file)

    results = [result for result in comparison["results"] if "error" not in result]
    if not results:
        raise RuntimeError("No successful model results found in comparison JSON.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_auc_accuracy_fps(
        results,
        args.output_dir / "metric_final_model_auc_accuracy_fps_kaixiang.png",
    )
    plot_distance_metric_auc(
        results,
        args.output_dir / "metric_final_euclidean_cosine_auc_kaixiang.png",
    )
    plot_thresholds(
        results,
        args.output_dir / "metric_final_thresholds_kaixiang.png",
    )


def model_labels(results):
    return [result["name"].replace("kaixiang_", "kx_").replace("zhongyu_", "zy_") for result in results]


def plot_auc_accuracy_fps(results, output_path):
    labels = model_labels(results)
    auc_values = [result["euclidean"]["roc_auc"] for result in results]
    accuracy_values = [result["euclidean"]["accuracy"] for result in results]
    fps_values = [result["fps"] for result in results]

    x_positions = list(range(len(results)))
    width = 0.35
    figure, left_axis = plt.subplots(figsize=(12, 6))

    left_axis.bar(
        [x - width / 2 for x in x_positions],
        auc_values,
        width=width,
        label="ROC-AUC",
    )
    left_axis.bar(
        [x + width / 2 for x in x_positions],
        accuracy_values,
        width=width,
        label="Accuracy",
    )
    left_axis.set_ylim(0, 1)
    left_axis.set_ylabel("Score")
    left_axis.set_xticks(x_positions)
    left_axis.set_xticklabels(labels, rotation=25, ha="right")
    left_axis.grid(axis="y", alpha=0.3)

    right_axis = left_axis.twinx()
    right_axis.plot(
        x_positions,
        fps_values,
        color="black",
        marker="o",
        linewidth=2,
        label="FPS",
    )
    right_axis.set_ylabel("FPS")

    for index, value in enumerate(auc_values):
        left_axis.text(index - width / 2, value + 0.01, f"{value:.3f}", ha="center", fontsize=8)
    for index, value in enumerate(accuracy_values):
        left_axis.text(index + width / 2, value + 0.01, f"{value:.3f}", ha="center", fontsize=8)
    for index, value in enumerate(fps_values):
        right_axis.text(index, value + 1.0, f"{value:.1f}", ha="center", fontsize=8)

    handles_left, labels_left = left_axis.get_legend_handles_labels()
    handles_right, labels_right = right_axis.get_legend_handles_labels()
    left_axis.legend(handles_left + handles_right, labels_left + labels_right, loc="lower right")
    figure.suptitle("Final Metric Recognition Model Comparison")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    print(f"Saved model comparison plot: {output_path}")


def plot_distance_metric_auc(results, output_path):
    labels = model_labels(results)
    euclidean_auc = [result["euclidean"]["roc_auc"] for result in results]
    cosine_auc = [result["cosine"]["roc_auc"] for result in results]
    x_positions = list(range(len(results)))
    width = 0.35

    plt.figure(figsize=(12, 5))
    plt.bar(
        [x - width / 2 for x in x_positions],
        euclidean_auc,
        width=width,
        label="Euclidean distance AUC",
    )
    plt.bar(
        [x + width / 2 for x in x_positions],
        cosine_auc,
        width=width,
        label="Cosine similarity AUC",
    )
    plt.ylim(0, 1)
    plt.ylabel("ROC-AUC")
    plt.xticks(x_positions, labels, rotation=25, ha="right")
    plt.title("Euclidean vs Cosine Similarity Metric Comparison")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved distance metric AUC plot: {output_path}")


def plot_thresholds(results, output_path):
    labels = model_labels(results)
    distance_thresholds = [
        result["euclidean"]["best_distance_threshold"] for result in results
    ]
    similarity_thresholds = [
        result["cosine"]["best_similarity_threshold"] for result in results
    ]
    x_positions = list(range(len(results)))

    figure, left_axis = plt.subplots(figsize=(12, 5))
    left_axis.plot(
        x_positions,
        distance_thresholds,
        marker="o",
        linewidth=2,
        label="Best Euclidean distance threshold",
    )
    left_axis.set_ylabel("Euclidean distance threshold")
    left_axis.set_xticks(x_positions)
    left_axis.set_xticklabels(labels, rotation=25, ha="right")
    left_axis.grid(axis="y", alpha=0.3)

    right_axis = left_axis.twinx()
    right_axis.plot(
        x_positions,
        similarity_thresholds,
        marker="s",
        linestyle="--",
        color="tab:orange",
        linewidth=2,
        label="Best cosine similarity threshold",
    )
    right_axis.set_ylabel("Cosine similarity threshold")

    handles_left, labels_left = left_axis.get_legend_handles_labels()
    handles_right, labels_right = right_axis.get_legend_handles_labels()
    left_axis.legend(handles_left + handles_right, labels_left + labels_right, loc="best")
    figure.suptitle("Verification Decision Thresholds")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    print(f"Saved threshold plot: {output_path}")


if __name__ == "__main__":
    main()
