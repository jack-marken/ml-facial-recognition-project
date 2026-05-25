"""Webcam test for Zhongyu's temporal liveness innovation feature."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from anti_spoofing.liveness_zhongyu import DEFAULT_MODEL_PATH, DEFAULT_THRESHOLD
from anti_spoofing.temporal_liveness_zhongyu import TemporalLivenessDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run detection + temporal liveness webcam test."
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to the trained liveness model.",
    )
    parser.add_argument(
        "--real-threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Temporal REAL probability threshold.",
    )
    parser.add_argument(
        "--spoof-threshold",
        type=float,
        default=None,
        help="Temporal SPOOF probability threshold. Defaults to real-threshold - 0.08.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=15,
        help="Number of recent frames used for temporal analysis.",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=5,
        help="Minimum frames required before making a temporal decision.",
    )
    parser.add_argument(
        "--confirmations",
        type=int,
        default=3,
        help="Consecutive proposed decisions required before switching label.",
    )
    parser.add_argument(
        "--max-stable-std",
        type=float,
        default=0.18,
        help="Maximum allowed standard deviation in the temporal window.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="OpenCV camera index.",
    )
    parser.add_argument(
        "--show-debug",
        action="store_true",
        help="Show raw temporal probability and stability diagnostics.",
    )
    parser.add_argument(
        "--process-every",
        type=int,
        default=3,
        help="Run detection and liveness inference every N frames.",
    )
    parser.add_argument(
        "--warmup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Warm up the liveness model before opening the display window.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Liveness model not found: {model_path}")

    temporal_detector = TemporalLivenessDetector(
        model_path=model_path,
        real_threshold=args.real_threshold,
        spoof_threshold=args.spoof_threshold,
        window_size=args.window_size,
        min_frames=args.min_frames,
        required_confirmations=args.confirmations,
        max_stable_std=args.max_stable_std,
    )
    if args.warmup:
        temporal_detector.update(np.zeros((224, 224, 3), dtype=np.uint8))
        temporal_detector.reset()

    from detection.detector import detect_and_crop_face

    webcam = cv2.VideoCapture(args.camera)
    if not webcam.isOpened():
        raise RuntimeError("Cannot open webcam.")
    webcam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    frame_index = 0
    process_every = max(1, args.process_every)
    last_detection_result = None
    last_temporal_result = None
    while webcam.isOpened():
        success, frame = webcam.read()
        if not success:
            break

        should_process = frame_index % process_every == 0
        frame_index += 1

        if should_process:
            detection_result = detect_and_crop_face(frame)
            if "face_image" in detection_result:
                last_detection_result = detection_result
                last_temporal_result = temporal_detector.update(
                    detection_result["face_image"]
                )
            else:
                temporal_detector.reset()
                last_detection_result = None
                last_temporal_result = None

        if last_detection_result is not None and last_temporal_result is not None:
            _draw_temporal_result(
                frame=frame,
                bbox=last_detection_result["bbox"],
                temporal_result=last_temporal_result,
                show_debug=args.show_debug,
            )
        else:
            cv2.putText(
                frame,
                "NO FACE",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (180, 180, 180),
                2,
            )

        cv2.imshow("Temporal Liveness Test", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    webcam.release()
    cv2.destroyAllWindows()


def _draw_temporal_result(
    frame,
    bbox,
    temporal_result: dict[str, float | int | str],
    show_debug: bool,
) -> None:
    x1, y1, x2, y2 = bbox
    label = str(temporal_result["liveness"])
    confidence = float(temporal_result["confidence"])
    color = _label_color(label)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    title_y = y1 - 34 if show_debug else y1 - 10
    debug_y = y1 - 10
    if title_y < 20:
        title_y = y1 + 24
        debug_y = y1 + 46

    cv2.putText(
        frame,
        f"{label} {confidence:.2f}",
        (x1, max(20, title_y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
    )
    if show_debug:
        debug_label = (
            f"raw {float(temporal_result['raw_real_probability']):.2f} | "
            f"temp {float(temporal_result['temporal_probability']):.2f} | "
            f"std {float(temporal_result['temporal_std']):.2f}"
        )
        cv2.putText(
            frame,
            debug_label,
            (x1, max(20, debug_y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )


def _label_color(label: str) -> tuple[int, int, int]:
    if label == "REAL":
        return (0, 210, 0)
    if label == "SPOOF":
        return (0, 0, 255)
    return (0, 190, 255)


if __name__ == "__main__":
    main()
