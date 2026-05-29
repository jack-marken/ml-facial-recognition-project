"""Small webcam smoke test for liveness integration.

This file is only for local testing. It uses the shared detection crop API and
then calls predict_liveness(face_image), matching the final integration flow.
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import cv2

from anti_spoofing.liveness_zhongyu import (
    DEFAULT_MODEL_PATH,
    DEFAULT_THRESHOLD,
    format_liveness_result,
    predict_liveness_probability,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run detection + liveness webcam test.")
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to the trained liveness .keras model.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="REAL probability threshold.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=15,
        help="Number of recent frames used for smoothing.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="OpenCV camera index.",
    )
    parser.add_argument(
        "--show-raw",
        action="store_true",
        help="Show current-frame raw REAL probability beside the smoothed result.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Liveness model not found: {model_path}")

    real_probability_window: deque[float] = deque(maxlen=max(1, args.window_size))

    from detection.detector import detect_and_crop_face

    webcam = cv2.VideoCapture(args.camera)
    if not webcam.isOpened():
        raise RuntimeError("Cannot open webcam.")

    while webcam.isOpened():
        success, frame = webcam.read()
        if not success:
            break

        detection_result = detect_and_crop_face(frame)
        if "face_image" in detection_result:
            face_image = detection_result["face_image"]
            real_probability = predict_liveness_probability(
                face_image,
                model_path=model_path,
            )
            real_probability_window.append(real_probability)
            smoothed_probability = sum(real_probability_window) / len(
                real_probability_window
            )
            liveness_result = format_liveness_result(
                smoothed_probability,
                threshold=args.threshold,
            )

            x1, y1, x2, y2 = detection_result["bbox"]
            label = f"{liveness_result['liveness']} {liveness_result['confidence']:.2f}"
            if args.show_raw:
                label = f"{label} (raw {real_probability:.2f})"
            color = (0, 255, 0) if liveness_result["liveness"] == "REAL" else (0, 0, 255)
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
            real_probability_window.clear()

        cv2.imshow("Liveness Smoke Test", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    webcam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
