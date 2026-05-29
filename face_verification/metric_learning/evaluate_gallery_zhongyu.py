"""Evaluate a recognition gallery against labelled face folders."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from .embedding_model_zhongyu import DEFAULT_ARCHITECTURE
from .recognition_zhongyu import (
    DEFAULT_GALLERY_PATH,
    calculate_identity_scores,
    format_identity_result,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate gallery matching accuracy.")
    parser.add_argument("--db-path", default="datasets/faces_db")
    parser.add_argument("--gallery", default=str(DEFAULT_GALLERY_PATH))
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--architecture",
        default=DEFAULT_ARCHITECTURE,
        choices=["efficientnet_b0", "resnet34"],
    )
    parser.add_argument("--threshold", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    db_path = Path(args.db_path)
    total = 0
    correct = 0

    for identity_dir in sorted(path for path in db_path.iterdir() if path.is_dir()):
        for image_path in sorted(identity_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            face_image = _load_resized_face(image_path)
            scores = calculate_identity_scores(
                face_image,
                gallery_path=args.gallery,
                model_path=args.model,
                architecture=args.architecture,
            )
            result = format_identity_result(scores, threshold=args.threshold)
            predicted = result.get("best_identity", result["identity"])
            score = result.get("best_similarity_score", result["similarity_score"])
            is_correct = predicted == identity_dir.name
            total += 1
            correct += int(is_correct)
            mark = "OK" if is_correct else "MISS"
            print(f"{mark} expected={identity_dir.name} predicted={predicted} score={score:.4f} file={image_path.name}")

    accuracy = correct / max(total, 1)
    print(f"Accuracy: {correct}/{total} = {accuracy:.4f}")


def _load_resized_face(image_path: Path):
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise ValueError(f"Failed to read image: {image_path}")
    resized_bgr = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)


if __name__ == "__main__":
    main()
