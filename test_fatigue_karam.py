"""Webcam test for Karam's fatigue detection (Yawn Eye Dataset CNN).

Run from repo root:
    python test_fatigue_karam.py

Controls:  Q = quit

Overlay shows:
  Indicator  — what the model sees: Closed / Open / Yawn / no_yawn
  Confidence — model certainty (0-100%)
  PERCLOS    — % of last 30 frames flagged as drowsy
  Status     — ALERT (green) or DROWSY (orange)

Model needed: models/fatigue_karam.h5
  Train it first with:
  python -m fatigue_detection.train_fatigue_karam
"""

import sys
import cv2
import numpy as np
from pathlib import Path

from detection.detector import detect_and_crop_face
from fatigue_detection import FatigueDetector

MODEL_PATH = Path("models/fatigue_karam.h5")


def main():
    model_ready = MODEL_PATH.exists()

    detector = FatigueDetector(model_path=str(MODEL_PATH))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        sys.exit(1)

    print("Fatigue Detection Test — Karam")
    print("  Indicator classes: Closed | Open | Yawn | no_yawn")
    print("  Press Q to quit\n")

    if not model_ready:
        print("  [!] Model not found — train it first:")
        print("      python -m fatigue_detection.train_fatigue_karam\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detection = detect_and_crop_face(frame)

        if "face_image" not in detection:
            cv2.putText(frame, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 2)
        else:
            face = detection["face_image"]
            x1, y1, x2, y2 = detection["bbox"]

            if not model_ready:
                # Show placeholder until model is trained
                cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 180, 180), 2)
                cv2.putText(frame, "Model not trained yet", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 80, 255), 2)
                cv2.putText(frame, "Run: python -m fatigue_detection.train_fatigue_karam",
                            (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            else:
                result = detector.update(face)

                is_drowsy  = result["fatigue"] == "DROWSY"
                box_colour = (0, 140, 255) if is_drowsy else (0, 210, 0)
                indicator  = result["indicator"]

                # Determine indicator colour
                if indicator in ("Closed", "Yawn"):
                    ind_colour = (0, 140, 255)   # orange = fatigue signal
                else:
                    ind_colour = (0, 210, 0)      # green  = alert signal

                cv2.rectangle(frame, (x1, y1), (x2, y2), box_colour, 2)

                lines = [
                    (f"Status     : {result['fatigue']}",        box_colour),
                    (f"Indicator  : {indicator}",                 ind_colour),
                    (f"Confidence : {result['confidence']*100:.1f}%", (230, 230, 230)),
                    (f"PERCLOS    : {result['perclos']*100:.1f}%",    (230, 230, 230)),
                ]
                for i, (text, colour) in enumerate(lines):
                    y = 30 + i * 28
                    cv2.putText(frame, text, (10, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
                    cv2.putText(frame, text, (10, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, 2)

        cv2.imshow("Fatigue Detection — Karam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
