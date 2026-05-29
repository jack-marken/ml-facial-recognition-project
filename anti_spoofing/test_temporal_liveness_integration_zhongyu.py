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
        "--min-motion-for-real",
        type=float,
        default=0.004,
        help="Minimum average face-crop motion required before confirming REAL.",
    )
    parser.add_argument(
        "--disable-motion-check",
        action="store_true",
        help="Disable the static-face motion guard.",
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
        default=5,
        help="Run detection and liveness inference every N frames.",
    )
    parser.add_argument(
        "--reset-iou-threshold",
        type=float,
        default=0.35,
        help="Reset temporal state when the new face box IoU drops below this value.",
    )
    parser.add_argument(
        "--reset-center-shift",
        type=float,
        default=0.25,
        help="Reset temporal state when face-box center shift is large relative to box size.",
    )
    parser.add_argument(
        "--reset-area-change",
        type=float,
        default=0.45,
        help="Reset temporal state when face-box area changes by this ratio.",
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
        min_motion_for_real=args.min_motion_for_real,
        enable_motion_check=not args.disable_motion_check,
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
    last_processed_bbox = None
    while webcam.isOpened():
        success, frame = webcam.read()
        if not success:
            break

        should_process = frame_index % process_every == 0
        frame_index += 1

        if should_process:
            detection_result = detect_and_crop_face(frame)
            if "face_image" in detection_result:
                face_image = detection_result["face_image"]
                current_bbox = detection_result["bbox"]

                if last_processed_bbox is not None and _is_new_face_track(
                    previous_bbox=last_processed_bbox,
                    current_bbox=current_bbox,
                    iou_threshold=args.reset_iou_threshold,
                    center_shift_threshold=args.reset_center_shift,
                    area_change_threshold=args.reset_area_change,
                ):
                    temporal_detector.reset()
                    last_temporal_result = None

                last_detection_result = detection_result
                last_processed_bbox = current_bbox
                last_temporal_result = temporal_detector.update(face_image)
            else:
                temporal_detector.reset()
                last_detection_result = None
                last_temporal_result = None
                last_processed_bbox = None

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
    title_y = y1 - 56 if show_debug else y1 - 10
    debug_y = y1 - 32
    motion_y = y1 - 10
    if title_y < 20:
        title_y = y1 + 24
        debug_y = y1 + 46
        motion_y = y1 + 68

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
        motion_label = (
            f"motion {float(temporal_result['motion_score']):.4f} | "
            f"active {temporal_result['motion_enough']}"
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
        cv2.putText(
            frame,
            motion_label,
            (x1, max(20, motion_y)),
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


def _is_new_face_track(
    previous_bbox,
    current_bbox,
    iou_threshold: float,
    center_shift_threshold: float,
    area_change_threshold: float,
) -> bool:
    previous_area = _bbox_area(previous_bbox)
    current_area = _bbox_area(current_bbox)
    if previous_area <= 0 or current_area <= 0:
        return True

    iou = _bbox_iou(previous_bbox, current_bbox)
    center_shift = _normalized_center_shift(previous_bbox, current_bbox)
    area_change = abs(current_area - previous_area) / max(previous_area, current_area)

    return (
        iou < iou_threshold
        or center_shift > center_shift_threshold
        or area_change > area_change_threshold
    )


def _bbox_area(bbox) -> float:
    x1, y1, x2, y2 = bbox
    return float(max(0, x2 - x1) * max(0, y2 - y1))


def _bbox_iou(first_bbox, second_bbox) -> float:
    first_x1, first_y1, first_x2, first_y2 = first_bbox
    second_x1, second_y1, second_x2, second_y2 = second_bbox

    inter_x1 = max(first_x1, second_x1)
    inter_y1 = max(first_y1, second_y1)
    inter_x2 = min(first_x2, second_x2)
    inter_y2 = min(first_y2, second_y2)
    intersection = _bbox_area([inter_x1, inter_y1, inter_x2, inter_y2])
    union = _bbox_area(first_bbox) + _bbox_area(second_bbox) - intersection
    return intersection / union if union > 0 else 0.0


def _normalized_center_shift(first_bbox, second_bbox) -> float:
    first_x1, first_y1, first_x2, first_y2 = first_bbox
    second_x1, second_y1, second_x2, second_y2 = second_bbox
    first_center_x = (first_x1 + first_x2) / 2
    first_center_y = (first_y1 + first_y2) / 2
    second_center_x = (second_x1 + second_x2) / 2
    second_center_y = (second_y1 + second_y2) / 2
    center_distance = np.hypot(
        second_center_x - first_center_x,
        second_center_y - first_center_y,
    )
    average_box_size = (
        (first_x2 - first_x1)
        + (first_y2 - first_y1)
        + (second_x2 - second_x1)
        + (second_y2 - second_y1)
    ) / 4
    return center_distance / max(average_box_size, 1.0)


if __name__ == "__main__":
    main()
