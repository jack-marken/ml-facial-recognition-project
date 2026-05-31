import csv
import json
import math
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUT_DIR = Path("reports/kaixiang_individual_report")
GROUP_RESULTS = Path("reports/kaixiang_zhongyu_recomparison/comparison_results.json")
GROUP_DIR = Path("reports/kaixiang_zhongyu_recomparison")


GLASSES_RESULTS = [
    {
        "name": "glasses_mobilenetv2",
        "model": "MobileNetV2 + binary head",
        "accuracy": 0.9931872037914692,
        "precision": 0.9891540130151844,
        "recall": 0.9941860465116279,
        "f1": 0.9916636462486408,
        "loss": 0.024645544499470366,
        "fps": 119.1220690805163,
        "samples": 3376,
        "tp": 1368,
        "tn": 1985,
        "fp": 15,
        "fn": 8,
    },
    {
        "name": "glasses_efficientnetb0",
        "model": "EfficientNetB0 + binary head",
        "accuracy": 0.9922985781990521,
        "precision": 0.9884225759768451,
        "recall": 0.9927325581395349,
        "f1": 0.9905728788977519,
        "loss": 0.025231964657133774,
        "fps": 103.51579310344135,
        "samples": 3376,
        "tp": 1366,
        "tn": 1984,
        "fp": 16,
        "fn": 10,
    },
]


def font(size=22, bold=False):
    names = ["arialbd.ttf" if bold else "arial.ttf", "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


FONT_TITLE = font(34, True)
FONT_LABEL = font(22)
FONT_SMALL = font(18)
FONT_TINY = font(15)


def short_name(name):
    return (
        name.replace("kaixiang_", "")
        .replace("contrastive_", "Cont. ")
        .replace("triplet_", "Triplet ")
        .replace("mobilenetv2", "MobileNetV2")
        .replace("resnet18", "ResNet18")
        .replace("efficientnetb0", "EfficientNetB0")
        .replace("glasses_", "")
    )


def load_results():
    data = json.loads(GROUP_RESULTS.read_text(encoding="utf-8"))
    recognition = [row for row in data["recognition"] if row["owner"] == "Kaixiang"]
    liveness = [row for row in data["liveness"] if row["owner"] == "Kaixiang"]
    return recognition, liveness


def write_csv(path, headers, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in headers})


def draw_axes(draw, box, y_label="Score"):
    left, top, right, bottom = box
    draw.line((left, top, left, bottom), fill="black", width=2)
    draw.line((left, bottom, right, bottom), fill="black", width=2)
    for i in range(6):
        y = bottom - (bottom - top) * i / 5
        value = i / 5
        draw.line((left - 6, y, right, y), fill=(220, 220, 220), width=1)
        draw.text((left - 58, y - 10), f"{value:.1f}", fill="black", font=FONT_TINY)
    draw.text((left - 80, top - 35), y_label, fill="black", font=FONT_SMALL)


def draw_metric_bars(rows, output_path, title, series):
    width, height = 1500, 760
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((width // 2, 35), title, anchor="mm", fill="black", font=FONT_TITLE)
    plot = (110, 120, 1270, 600)
    draw_axes(draw, plot)
    left, top, right, bottom = plot
    colors = [(43, 120, 180), (255, 127, 14), (44, 160, 44)]
    n = len(rows)
    group_w = (right - left) / n
    bar_w = group_w / (len(series) + 1.7)
    for i, row in enumerate(rows):
        cx = left + group_w * (i + 0.5)
        for j, (label, key) in enumerate(series):
            value = float(row[key])
            x0 = cx - (len(series) * bar_w) / 2 + j * bar_w
            x1 = x0 + bar_w * 0.8
            y0 = bottom - (bottom - top) * value
            draw.rectangle((x0, y0, x1, bottom), fill=colors[j])
            draw.text(((x0 + x1) / 2, y0 - 18), f"{value:.3f}", anchor="mm", fill="black", font=FONT_TINY)
        label_text = row["model"]
        draw.text((cx, bottom + 18), label_text, anchor="ma", fill="black", font=FONT_TINY)
    legend_x, legend_y = 1300, 160
    for j, (label, _) in enumerate(series):
        y = legend_y + j * 40
        draw.rectangle((legend_x, y, legend_x + 28, y + 20), fill=colors[j])
        draw.text((legend_x + 40, y - 2), label, fill="black", font=FONT_SMALL)
    image.save(output_path)


def draw_bar_with_fps(rows, output_path, title):
    width, height = 1500, 800
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((width // 2, 35), title, anchor="mm", fill="black", font=FONT_TITLE)
    plot = (110, 125, 1270, 610)
    draw_axes(draw, plot)
    left, top, right, bottom = plot
    n = len(rows)
    group_w = (right - left) / n
    bar_w = group_w / 4.1
    colors = {"auc": (43, 120, 180), "accuracy": (255, 127, 14), "f1": (44, 160, 44)}
    max_fps = max(float(row["fps"]) for row in rows) * 1.1
    points = []
    for i, row in enumerate(rows):
        cx = left + group_w * (i + 0.5)
        for j, key in enumerate(["auc", "accuracy", "f1"]):
            value = float(row[key])
            x0 = cx - 1.5 * bar_w + j * bar_w
            x1 = x0 + bar_w * 0.8
            y0 = bottom - (bottom - top) * value
            draw.rectangle((x0, y0, x1, bottom), fill=colors[key])
        fps_y = bottom - (bottom - top) * float(row["fps"]) / max_fps
        points.append((cx + bar_w * 1.55, fps_y))
        draw.text((cx, bottom + 18), row["model"], anchor="ma", fill="black", font=FONT_TINY)
    for a, b in zip(points, points[1:]):
        draw.line((*a, *b), fill="black", width=3)
    for x, y in points:
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill="black")
    legend_x, legend_y = 1300, 155
    for i, (label, color) in enumerate([("ROC-AUC", colors["auc"]), ("Accuracy", colors["accuracy"]), ("F1", colors["f1"]), ("FPS", "black")]):
        y = legend_y + i * 40
        if label == "FPS":
            draw.line((legend_x, y + 10, legend_x + 28, y + 10), fill="black", width=3)
            draw.ellipse((legend_x + 9, y + 4, legend_x + 19, y + 14), fill="black")
        else:
            draw.rectangle((legend_x, y, legend_x + 28, y + 20), fill=color)
        draw.text((legend_x + 40, y - 2), label, fill="black", font=FONT_SMALL)
    draw.text((1290, 585), "FPS line scaled to its own max", fill=(80, 80, 80), font=FONT_TINY)
    image.save(output_path)


def draw_roc(rows, output_path, title, metric_key="euclidean"):
    width, height = 1450, 820
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((width // 2, 36), title, anchor="mm", fill="black", font=FONT_TITLE)
    left, top, right, bottom = 110, 105, 845, 700
    draw.line((left, top, left, bottom), fill="black", width=2)
    draw.line((left, bottom, right, bottom), fill="black", width=2)
    for i in range(6):
        x = left + (right - left) * i / 5
        y = bottom - (bottom - top) * i / 5
        draw.line((x, top, x, bottom), fill=(225, 225, 225), width=1)
        draw.line((left, y, right, y), fill=(225, 225, 225), width=1)
        draw.text((x, bottom + 12), f"{i/5:.1f}", anchor="ma", fill="black", font=FONT_TINY)
        draw.text((left - 28, y), f"{i/5:.1f}", anchor="rm", fill="black", font=FONT_TINY)
    draw.line((left, bottom, right, top), fill=(130, 130, 130), width=2)
    colors = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40)]
    for idx, row in enumerate(rows):
        metric = row[metric_key]
        pts = []
        for fpr, tpr in zip(metric["fpr"], metric["tpr"]):
            x = left + (right - left) * fpr
            y = bottom - (bottom - top) * tpr
            pts.append((x, y))
        if len(pts) > 1:
            draw.line(pts, fill=colors[idx % len(colors)], width=4)
        lx, ly = 890, 140 + idx * 50
        draw.line((lx, ly, lx + 34, ly), fill=colors[idx % len(colors)], width=4)
        draw.text((lx + 44, ly - 12), f"{short_name(row['name'])} AUC={metric['roc_auc']:.3f}", fill="black", font=FONT_SMALL)
    draw.text(((left + right) / 2, 745), "False Positive Rate", anchor="mm", fill="black", font=FONT_LABEL)
    draw.text((45, (top + bottom) / 2), "TPR", anchor="mm", fill="black", font=FONT_LABEL)
    image.save(output_path)


def draw_distance_metric(best, output_path):
    rows = [
        {"model": "Euclidean", "auc": best["euclidean"]["roc_auc"], "accuracy": best["euclidean"]["accuracy"]},
        {"model": "Cosine", "auc": best["cosine"]["roc_auc"], "accuracy": best["cosine"]["accuracy"]},
    ]
    draw_metric_bars(rows, output_path, "Best Metric Model: Euclidean vs Cosine", [("ROC-AUC", "auc"), ("Accuracy", "accuracy")])


def draw_confusion_matrix(row, output_path, title):
    matrix = [[row["tn"], row["fp"]], [row["fn"], row["tp"]]]
    labels = [["TN", "FP"], ["FN", "TP"]]
    max_value = max(max(r) for r in matrix)
    image = Image.new("RGB", (760, 650), "white")
    draw = ImageDraw.Draw(image)
    draw.text((380, 35), title, anchor="mm", fill="black", font=FONT_TITLE)
    x0, y0, cell = 160, 130, 190
    col_names = ["Pred without", "Pred with"]
    row_names = ["True without", "True with"]
    for i, name in enumerate(col_names):
        draw.text((x0 + i * cell + cell / 2, y0 - 35), name, anchor="mm", fill="black", font=FONT_SMALL)
    for j, name in enumerate(row_names):
        draw.text((x0 - 70, y0 + j * cell + cell / 2), name, anchor="mm", fill="black", font=FONT_SMALL)
    for r in range(2):
        for c in range(2):
            value = matrix[r][c]
            shade = int(245 - 150 * value / max_value)
            draw.rectangle((x0 + c * cell, y0 + r * cell, x0 + (c + 1) * cell, y0 + (r + 1) * cell), fill=(shade, shade, 255), outline="black", width=2)
            draw.text((x0 + c * cell + cell / 2, y0 + r * cell + cell / 2 - 15), labels[r][c], anchor="mm", fill="black", font=FONT_LABEL)
            draw.text((x0 + c * cell + cell / 2, y0 + r * cell + cell / 2 + 24), str(value), anchor="mm", fill="black", font=FONT_TITLE)
    image.save(output_path)


def build_summary(recognition, liveness, glasses):
    best_metric = max(recognition, key=lambda row: row["euclidean"]["roc_auc"])
    best_liveness = max(liveness, key=lambda row: row["roc_auc"])
    best_glasses = max(glasses, key=lambda row: row["f1"])
    lines = [
        "# Kaixiang Individual Report Assets",
        "",
        "This folder contains comparison tables and figures for Kaixiang's individual report only.",
        "",
        f"Best metric-recognition model: **{short_name(best_metric['name'])}** (Euclidean ROC-AUC={best_metric['euclidean']['roc_auc']:.4f}).",
        f"Best liveness model: **{short_name(best_liveness['name'])}** (ROC-AUC={best_liveness['roc_auc']:.4f}).",
        f"Best glasses model: **{best_glasses['model']}** (F1={best_glasses['f1']:.4f}).",
        "",
        "Recommended figures for a <=5 page individual report:",
        "- metric_kaixiang_model_comparison.png",
        "- metric_kaixiang_roc_curves.png",
        "- metric_best_score_distribution.png",
        "- liveness_kaixiang_model_comparison.png",
        "- liveness_best_score_distribution.png",
        "- glasses_kaixiang_model_comparison.png",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    recognition, liveness = load_results()

    recognition_rows = []
    for row in recognition:
        loss = "Contrastive loss" if "contrastive" in row["name"] else "Triplet loss"
        recognition_rows.append(
            {
                "name": row["name"],
                "model": short_name(row["name"]),
                "loss": loss,
                "euc_auc": row["euclidean"]["roc_auc"],
                "euc_acc": row["euclidean"]["accuracy"],
                "euc_f1": row["euclidean"]["f1"],
                "cos_auc": row["cosine"]["roc_auc"],
                "cos_acc": row["cosine"]["accuracy"],
                "fps": row["fps"],
            }
        )
    write_csv(
        OUT_DIR / "metric_kaixiang_comparison.csv",
        ["model", "loss", "euc_auc", "euc_acc", "euc_f1", "cos_auc", "cos_acc", "fps"],
        recognition_rows,
    )

    liveness_rows = [
        {
            "name": row["name"],
            "model": short_name(row["name"]),
            "loss": "Binary cross-entropy",
            "auc": row["roc_auc"],
            "accuracy": row["accuracy"],
            "f1": row["f1"],
            "precision": row["precision"],
            "recall": row["recall"],
            "fps": row["fps"],
        }
        for row in liveness
    ]
    write_csv(
        OUT_DIR / "liveness_kaixiang_comparison.csv",
        ["model", "loss", "auc", "accuracy", "f1", "precision", "recall", "fps"],
        liveness_rows,
    )

    write_csv(
        OUT_DIR / "glasses_kaixiang_comparison.csv",
        ["model", "loss", "accuracy", "precision", "recall", "f1", "fps", "samples"],
        [{**row, "loss": "Binary cross-entropy"} for row in GLASSES_RESULTS],
    )

    draw_bar_with_fps(
        [
            {"model": row["model"], "auc": row["euc_auc"], "accuracy": row["euc_acc"], "f1": row["euc_f1"], "fps": row["fps"]}
            for row in recognition_rows
        ],
        OUT_DIR / "metric_kaixiang_model_comparison.png",
        "Kaixiang Metric Learning Model Comparison",
    )
    draw_roc(recognition, OUT_DIR / "metric_kaixiang_roc_curves.png", "Kaixiang Face Verification ROC Curves")
    best_metric = max(recognition, key=lambda row: row["euclidean"]["roc_auc"])
    draw_distance_metric(best_metric, OUT_DIR / "metric_best_distance_metric_comparison.png")

    draw_bar_with_fps(
        liveness_rows,
        OUT_DIR / "liveness_kaixiang_model_comparison.png",
        "Kaixiang Liveness Model Comparison",
    )

    draw_bar_with_fps(
        [
            {"model": row["model"], "auc": row["f1"], "accuracy": row["accuracy"], "f1": row["f1"], "fps": row["fps"]}
            for row in GLASSES_RESULTS
        ],
        OUT_DIR / "glasses_kaixiang_model_comparison.png",
        "Kaixiang Glasses Detection Model Comparison",
    )
    draw_confusion_matrix(
        GLASSES_RESULTS[0],
        OUT_DIR / "glasses_mobilenetv2_confusion_matrix.png",
        "Glasses Detection Confusion Matrix",
    )

    source_metric_dist = GROUP_DIR / "recognition_best_model_score_distribution.png"
    if source_metric_dist.exists():
        shutil.copyfile(source_metric_dist, OUT_DIR / "metric_best_score_distribution.png")
    source_liveness_dist = GROUP_DIR / "liveness_best_model_score_distribution.png"
    if source_liveness_dist.exists():
        shutil.copyfile(source_liveness_dist, OUT_DIR / "liveness_best_score_distribution.png")

    build_summary(recognition, liveness, GLASSES_RESULTS)
    print(f"Saved Kaixiang individual report assets to: {OUT_DIR}")


if __name__ == "__main__":
    main()
