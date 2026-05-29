import argparse
from pathlib import Path

import cv2

from glasses_detection.glasses_kaixiang import DEFAULT_MODEL_PATH, DEFAULT_THRESHOLD, predict_glasses


def main():
    parser = argparse.ArgumentParser(description="Webcam test for detection + Kaixiang glasses detection.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Glasses model not found: {args.checkpoint}")

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
            result = predict_glasses(
                detection_result["face_image"],
                model_path=args.checkpoint,
                threshold=args.threshold,
            )
            x1, y1, x2, y2 = detection_result["bbox"]
            label = result["label"]
            probability = result["with_glasses_probability"]
            confidence = result["confidence"]
            color = (0, 255, 0) if label == "with_glasses" else (255, 180, 0)
            display = f"{label} conf={confidence:.2f} p_glasses={probability:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                display,
                (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                color,
                2,
            )
            print(result, end="\r")

        cv2.imshow("Kaixiang Glasses Detection Test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    webcam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

