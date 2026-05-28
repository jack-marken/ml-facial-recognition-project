import argparse
from pathlib import Path

import cv2

from face_verification.metric_learning.recognition_kaixiang import (
    DEFAULT_DISTANCE_THRESHOLD,
    DEFAULT_GALLERY_PATH,
    DEFAULT_MODEL_PATH,
    predict_identity,
)


def main():
    parser = argparse.ArgumentParser(
        description="Webcam test for detection + Kaixiang Siamese recognition."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--gallery", type=Path, default=DEFAULT_GALLERY_PATH)
    parser.add_argument("--distance-threshold", type=float, default=DEFAULT_DISTANCE_THRESHOLD)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Recognition model not found: {args.checkpoint}")
    if not args.gallery.exists():
        raise FileNotFoundError(f"Recognition gallery not found: {args.gallery}")

    from detection.detector import detect_and_crop_face

    webcam = cv2.VideoCapture(args.camera)
    if not webcam.isOpened():
        raise RuntimeError(f"Cannot open webcam index {args.camera}")

    print("Press q to quit.")
    while True:
        ok, frame = webcam.read()
        if not ok:
            break

        detection_result = detect_and_crop_face(frame)
        if "face_image" in detection_result:
            result = predict_identity(
                detection_result["face_image"],
                gallery_path=args.gallery,
                model_path=args.checkpoint,
                distance_threshold=args.distance_threshold,
            )
            x1, y1, x2, y2 = detection_result["bbox"]
            identity = result["identity"]
            score = result["similarity_score"]
            color = (0, 255, 0) if identity != "UNKNOWN" else (0, 0, 255)
            label = f"{identity} {score:.2f}"
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

        cv2.imshow("Kaixiang Recognition Integration Test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    webcam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
