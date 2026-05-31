import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


DEFAULT_INPUT = Path("reports/glasses_model_comparison_kaixiang.json")
DEFAULT_OUTPUT = Path("reports/glasses_model_comparison_no_auc_kaixiang.png")


def load_font(size: int, bold: bool = False):
    candidates = (
        ["arialbd.ttf", "Arial Bold.ttf"] if bold else ["arial.ttf", "Arial.ttf"]
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


FONT_TITLE = load_font(40, bold=True)
FONT_LABEL = load_font(24)
FONT_SMALL = load_font(20)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot glasses detection Accuracy/F1/FPS comparison without ROC-AUC."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def short_name(name: str) -> str:
    return (
        name.replace("glasses_", "")
        .replace("mobilenetv2", "MobileNetV2")
        .replace("efficientnetb0", "EfficientNetB0")
    )


def draw_chart(results, output_path: Path):
    width, height = 1500, 800
    left, right, top, bottom = 110, 260, 125, 190
    plot_width = width - left - right
    plot_height = height - top - bottom
    axis_bottom = top + plot_height

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    draw.text(
        (width // 2, 48),
        "Kaixiang Glasses Detection Model Comparison",
        anchor="mm",
        fill="black",
        font=FONT_TITLE,
    )
    draw.text((30, 100), "Score", fill="black", font=FONT_LABEL)

    # Axes and score grid.
    draw.line((left, top, left, axis_bottom), fill="black", width=2)
    draw.line((left, axis_bottom, left + plot_width, axis_bottom), fill="black", width=2)
    for tick in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        y = axis_bottom - tick * plot_height
        draw.line((left - 6, y, left, y), fill="black", width=2)
        draw.text((left - 16, y), f"{tick:.1f}", anchor="rm", fill="black", font=FONT_SMALL)
        draw.line((left, y, left + plot_width, y), fill=(220, 220, 220), width=1)

    labels = [short_name(row["name"]) for row in results]
    accuracies = [row["accuracy"] for row in results]
    f1_scores = [row["f1"] for row in results]
    fps_values = [row["fps"] for row in results]
    max_fps = max(fps_values) if fps_values else 1.0

    group_width = plot_width / max(len(results), 1)
    bar_width = min(90, group_width * 0.20)
    colors = {
        "Accuracy": (255, 127, 14),
        "F1": (44, 160, 44),
    }

    fps_points = []
    for index, label in enumerate(labels):
        center_x = left + group_width * (index + 0.5)

        for metric_index, (metric_name, value) in enumerate(
            [("Accuracy", accuracies[index]), ("F1", f1_scores[index])]
        ):
            offset = (metric_index - 0.5) * (bar_width + 16)
            x0 = center_x + offset - bar_width / 2
            x1 = center_x + offset + bar_width / 2
            y0 = axis_bottom - value * plot_height
            draw.rectangle((x0, y0, x1, axis_bottom), fill=colors[metric_name])
            draw.text(
                ((x0 + x1) / 2, y0 - 10),
                f"{value:.4f}",
                anchor="mb",
                fill="black",
                font=FONT_SMALL,
            )

        fps_y = axis_bottom - (fps_values[index] / max_fps) * plot_height
        fps_points.append((center_x, fps_y))
        draw.ellipse((center_x - 7, fps_y - 7, center_x + 7, fps_y + 7), fill="black")
        draw.text(
            (center_x, fps_y - 14),
            f"{fps_values[index]:.1f}",
            anchor="mb",
            fill="black",
            font=FONT_SMALL,
        )

        draw.text(
            (center_x, axis_bottom + 25),
            label,
            anchor="ma",
            fill="black",
            font=FONT_SMALL,
        )

    for first, second in zip(fps_points, fps_points[1:]):
        draw.line((first[0], first[1], second[0], second[1]), fill="black", width=3)

    # Legend.
    legend_x, legend_y = width - right + 30, 155
    for i, (label, color) in enumerate(
        [("Accuracy", colors["Accuracy"]), ("F1", colors["F1"]), ("FPS", "black")]
    ):
        y = legend_y + i * 44
        if label == "FPS":
            draw.line((legend_x, y + 12, legend_x + 34, y + 12), fill="black", width=3)
            draw.ellipse((legend_x + 12, y + 5, legend_x + 22, y + 15), fill="black")
        else:
            draw.rectangle((legend_x, y, legend_x + 34, y + 24), fill=color)
        draw.text((legend_x + 48, y - 1), label, fill="black", font=FONT_LABEL)

    draw.text(
        (width - 210, height - 50),
        "FPS line scaled to its own max",
        anchor="mm",
        fill=(80, 80, 80),
        font=FONT_SMALL,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def main():
    args = parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    draw_chart(data["results"], args.output)
    print(f"Saved glasses comparison figure to: {args.output}")


if __name__ == "__main__":
    main()
