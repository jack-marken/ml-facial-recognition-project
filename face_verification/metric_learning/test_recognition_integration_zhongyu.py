"""Webcam smoke test for detection + metric-learning recognition."""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import cv2

from face_verification.metric_learning.embedding_model_zhongyu import DEFAULT_ARCHITECTURE
from face_verification.metric_learning.recognition_zhongyu import (
    DEFAULT_GALLERY_PATH,
    DEFAULT_THRESHOLD,
    calculate_identity_scores,
    format_identity_result,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run detection + recognition webcam test.")
    parser.add_argument("--gallery", default=str(DEFAULT_GALLERY_PATH))
    parser.add_argument("--model", default=None, help="Optional trained .pth checkpoint.")
    parser.add_argument(
        "--architecture",
        default=DEFAULT_ARCHITECTURE,
        choices=["efficientnet_b0", "resnet34"],
    )
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument(
        "--window-size",
        type=int,
        default=10,
        help="Number of recent frames used for smoothing identity scores.",
    )
    parser.add_argument(
        "--show-raw",
        action="store_true",
        help="Show current-frame raw best identity and score beside the smoothed result.",
    )
    parser.add_argument("--camera", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not Path(args.gallery).exists():
        raise FileNotFoundError(
            f"Recognition gallery not found: {args.gallery}. "
            "Run python -m face_verification.metric_learning.build_gallery_zhongyu first."
        )

    from detection.detector import detect_and_crop_face

    score_window: deque[dict[str, float]] = deque(maxlen=max(1, args.window_size))

    webcam = cv2.VideoCapture(args.camera)
    if not webcam.isOpened():
        raise RuntimeError("Cannot open webcam.")

    while webcam.isOpened():
        success, frame = webcam.read()
        if not success:
            break

        detection_result = detect_and_crop_face(frame)
        if "face_image" in detection_result:
            raw_scores = calculate_identity_scores(
                detection_result["face_image"],
                gallery_path=args.gallery,
                model_path=args.model,
                architecture=args.architecture,
            )
            score_window.append(raw_scores)
            recognition_result = format_identity_result(
                _average_identity_scores(score_window),
                threshold=args.threshold,
            )

            x1, y1, x2, y2 = detection_result["bbox"]
            identity = recognition_result["identity"]
            score = recognition_result["similarity_score"]
            color = (0, 255, 0) if identity != "UNKNOWN" else (0, 0, 255)
            label = f"{identity} {score:.2f}"
            if args.show_raw:
                raw_result = format_identity_result(raw_scores, threshold=args.threshold)
                label = (
                    f"{label} "
                    f"(raw {raw_result['identity']} {raw_result['similarity_score']:.2f})"
                )

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                label,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )
        else:
            score_window.clear()

        cv2.imshow("Recognition Smoke Test", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    webcam.release()
    cv2.destroyAllWindows()


def _average_identity_scores(score_window: deque[dict[str, float]]) -> dict[str, float]:
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}

    for scores in score_window:
        for identity, score in scores.items():
            totals[identity] = totals.get(identity, 0.0) + float(score)
            counts[identity] = counts.get(identity, 0) + 1

    return {
        identity: totals[identity] / counts[identity]
        for identity in totals
        if counts[identity] > 0
    }


if __name__ == "__main__":
    main()
