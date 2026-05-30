import argparse
from collections import Counter, deque
from pathlib import Path
import pickle

import cv2

from face_verification.metric_learning.recognition_kaixiang import (
    DEFAULT_DISTANCE_MARGIN,
    DEFAULT_DISTANCE_THRESHOLD,
    DEFAULT_GALLERY_PATH,
    DEFAULT_MATCHING_MODE,
    DEFAULT_MODEL_PATH,
    DEFAULT_TOP_K,
    predict_identity,
)


def main():
    parser = argparse.ArgumentParser(
        description="Webcam test for detection + Kaixiang Siamese recognition."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--gallery", type=Path, default=DEFAULT_GALLERY_PATH)
    parser.add_argument("--distance-threshold", type=float, default=DEFAULT_DISTANCE_THRESHOLD)
    parser.add_argument("--distance-margin", type=float, default=DEFAULT_DISTANCE_MARGIN)
    parser.add_argument("--matching-mode", choices=["mean", "sample", "topk"], default=DEFAULT_MATCHING_MODE)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--smooth-min-votes", type=int, default=3)
    parser.add_argument("--unknown-reset-frames", type=int, default=10)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Recognition model not found: {args.checkpoint}")
    if not args.gallery.exists():
        raise FileNotFoundError(f"Recognition gallery not found: {args.gallery}")
    warn_if_gallery_mismatch(args.gallery, args.checkpoint)

    from detection.detector import detect_and_crop_face

    webcam = cv2.VideoCapture(args.camera)
    if not webcam.isOpened():
        raise RuntimeError(f"Cannot open webcam index {args.camera}")

    print("Press q to quit.")
    prediction_window = deque(maxlen=max(1, args.smooth_window))
    stable_identity = "UNKNOWN"
    stable_votes = 0
    unknown_frames = 0
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
                distance_margin=args.distance_margin,
                matching_mode=args.matching_mode,
                top_k=args.top_k,
            )
            x1, y1, x2, y2 = detection_result["bbox"]
            identity = result["identity"]
            score = result["similarity_score"]
            best_identity = result.get("best_identity", identity)
            distance = result.get("distance", 0.0)
            if identity != "UNKNOWN":
                prediction_window.append(identity)
                unknown_frames = 0
                candidate_identity, candidate_votes = choose_stable_identity(
                    prediction_window,
                    min_votes=args.smooth_min_votes,
                )
                if candidate_identity != "UNKNOWN":
                    stable_identity = candidate_identity
                    stable_votes = candidate_votes
            else:
                unknown_frames += 1
                if unknown_frames >= args.unknown_reset_frames:
                    prediction_window.clear()
                    stable_identity = "UNKNOWN"
                    stable_votes = 0

            color = (0, 255, 0) if stable_identity != "UNKNOWN" else (0, 0, 255)
            label = (
                f"{stable_identity} votes={stable_votes}/{len(prediction_window)} "
                f"raw={identity} d={distance:.4f}"
            )
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
            print(
                f"identity={identity} best={best_identity} "
                f"d={distance:.4f} gap={result.get('distance_gap', 0.0):.4f} "
                f"ambiguous={result.get('ambiguous', False)} "
                f"top={result.get('top_matches', [])}",
                end="\r",
            )
        else:
            unknown_frames += 1
            if unknown_frames >= args.unknown_reset_frames:
                prediction_window.clear()
                stable_identity = "UNKNOWN"
                stable_votes = 0

        cv2.imshow("Kaixiang Recognition Integration Test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    webcam.release()
    cv2.destroyAllWindows()


def choose_stable_identity(prediction_window, min_votes):
    if not prediction_window:
        return "UNKNOWN", 0

    counts = Counter(prediction_window)
    identity, votes = counts.most_common(1)[0]
    if identity == "UNKNOWN" or votes < min_votes:
        return "UNKNOWN", votes
    return identity, votes


def warn_if_gallery_mismatch(gallery_path: Path, checkpoint_path: Path):
    with gallery_path.open("rb") as file:
        gallery = pickle.load(file)

    gallery_model_path = gallery.get("model_path")
    if not gallery_model_path:
        print("Warning: gallery has no saved model_path metadata. Rebuild it if recognition is unstable.")
        return

    resolved_gallery_model = Path(gallery_model_path).resolve()
    resolved_checkpoint = checkpoint_path.resolve()
    if resolved_gallery_model != resolved_checkpoint:
        print("Warning: gallery was built with a different checkpoint.")
        print(f"  Gallery model: {resolved_gallery_model}")
        print(f"  Current model: {resolved_checkpoint}")
        print("  Rebuild gallery with the same checkpoint before judging webcam accuracy.")


if __name__ == "__main__":
    main()
