import argparse
from pathlib import Path

import cv2

from anti_spoofing.liveness_kaixiang import (
    LivenessPredictorKaixiang,
    LivenessTemporalSmootherKaixiang,
)
from detection.detector import detect_and_crop_face


def crop_expanded_face(frame, bbox, expand_ratio):
    x1, y1, x2, y2 = bbox
    frame_height, frame_width = frame.shape[:2]
    box_width = x2 - x1
    box_height = y2 - y1

    pad_x = int(box_width * expand_ratio)
    pad_y = int(box_height * expand_ratio)

    expanded_x1 = max(0, x1 - pad_x)
    expanded_y1 = max(0, y1 - pad_y)
    expanded_x2 = min(frame_width, x2 + pad_x)
    expanded_y2 = min(frame_height, y2 + pad_y)

    expanded_crop = frame[expanded_y1:expanded_y2, expanded_x1:expanded_x2]
    if expanded_crop.size == 0:
        return None, [expanded_x1, expanded_y1, expanded_x2, expanded_y2]

    rgb_face = cv2.cvtColor(expanded_crop, cv2.COLOR_BGR2RGB)
    rgb_face = cv2.resize(rgb_face, (224, 224))
    return rgb_face, [expanded_x1, expanded_y1, expanded_x2, expanded_y2]


def main():
    parser = argparse.ArgumentParser(
        description="Webcam integration test for detection crop + Kaixiang liveness."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/liveness_mobilenetv2_kaixiang_best.pth"),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.81,
        help="Minimum REAL probability required to output REAL.",
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument(
        "--expand-ratio",
        type=float,
        default=0.327,
        help=(
            "Expand the detected face bbox before liveness prediction. "
            "Example: 0.3 keeps more spoof context around the face."
        ),
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="Number of recent frames used for liveness temporal smoothing.",
    )
    parser.add_argument(
        "--spoof-votes",
        type=int,
        default=3,
        help="Minimum SPOOF votes in the smoothing window required to display SPOOF.",
    )
    args = parser.parse_args()

    predictor = LivenessPredictorKaixiang(args.checkpoint, threshold=args.threshold)
    smoother = LivenessTemporalSmootherKaixiang(
        window_size=args.smooth_window,
        spoof_votes=args.spoof_votes,
    )
    webcam = cv2.VideoCapture(args.camera)

    if not webcam.isOpened():
        raise RuntimeError(f"Cannot open webcam index {args.camera}")

    print("Press q to quit.")
    while True:
        ok, frame = webcam.read()
        if not ok:
            break

        detection_result = detect_and_crop_face(frame)
        if "status" in detection_result:
            smoother.reset()
            cv2.putText(
                frame,
                detection_result["status"],
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
            )
        else:
            display_bbox = detection_result["bbox"]
            face_image = detection_result["face_image"]

            if args.expand_ratio > 0:
                expanded_face, expanded_bbox = crop_expanded_face(
                    frame,
                    detection_result["bbox"],
                    args.expand_ratio,
                )
                if expanded_face is not None:
                    face_image = expanded_face
                    display_bbox = expanded_bbox

            raw_result = predictor.predict(face_image)
            result = smoother.update(raw_result)
            x1, y1, x2, y2 = display_bbox
            label = (
                f"{result['liveness']} {result['confidence']:.2f} "
                f"(raw {result['raw_liveness']}, "
                f"S:{result['smooth_spoof_votes']}/R:{result['smooth_real_votes']})"
            )
            if result["liveness"] == "REAL":
                color = (0, 255, 0)
            elif result["liveness"] == "SPOOF":
                color = (0, 0, 255)
            else:
                color = (0, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                label,
                (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )

        cv2.imshow("Kaixiang Liveness Integration Test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    webcam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
