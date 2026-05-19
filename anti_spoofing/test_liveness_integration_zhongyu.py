"""Small webcam smoke test for liveness integration.

This file is only for local testing. It uses the shared detection crop API and
then calls predict_liveness(face_image), matching the final integration flow.
"""

from __future__ import annotations

import cv2

from detection.detector import detect_and_crop_face
from anti_spoofing.liveness_zhongyu import predict_liveness


def main() -> None:
    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        raise RuntimeError("Cannot open webcam.")

    while webcam.isOpened():
        success, frame = webcam.read()
        if not success:
            break

        detection_result = detect_and_crop_face(frame)
        if "face_image" in detection_result:
            face_image = detection_result["face_image"]
            liveness_result = predict_liveness(face_image)

            x1, y1, x2, y2 = detection_result["bbox"]
            label = (
                f"{liveness_result['liveness']} "
                f"{liveness_result['confidence']:.2f}"
            )
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

        cv2.imshow("Liveness Smoke Test", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    webcam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
