"""Build a face embedding gallery from a trained classification model (Karam).

Mirrors build_gallery_zhongyu.py so the two galleries are interchangeable.
Each registered identity is stored as the mean L2-normalised embedding of
all their training images.

Usage:
    python -m face_verification.classification_model.build_gallery_karam \\
        --db-path  datasets/faces_db \\
        --model    models/recognition_classification_karam.pth \\
        --output   models/recognition_gallery_classification_karam.pkl
"""
# Author: Karam

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np

from .classification_model_karam import generate_embedding, load_classification_model


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
DEFAULT_MODEL_PATH = Path("models/recognition_classification_karam.pth")
DEFAULT_GALLERY_PATH = Path("models/recognition_gallery_classification_karam.pkl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build classification-model face gallery (Karam)."
    )
    parser.add_argument("--db-path", default="datasets/faces_db")
    parser.add_argument("--model",   default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--output",  default=str(DEFAULT_GALLERY_PATH))
    parser.add_argument(
        "--skip-detection", action="store_true",
        help="Use direct resize instead of YOLO crop for registration images.",
    )
    return parser.parse_args()


def main() -> None:
    args     = parse_args()
    db_path  = Path(args.db_path)
    out_path = Path(args.output)

    if not db_path.exists():
        raise FileNotFoundError(f"Face database not found: {db_path}")

    model = load_classification_model(model_path=args.model)

    cropper = None
    if not args.skip_detection:
        from detection.detector import detect_and_crop_face
        cropper = detect_and_crop_face

    identity_embeddings: dict[str, np.ndarray] = {}
    sample_counts: dict[str, int]              = {}
    skipped: list[str]                         = []

    for identity_dir in sorted(p for p in db_path.iterdir() if p.is_dir()):
        embeddings: list[np.ndarray] = []
        for image_path in sorted(identity_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            face_image = _load_face_image(image_path, cropper=cropper)
            if face_image is None:
                skipped.append(str(image_path))
                continue

            embeddings.append(generate_embedding(model, face_image))

        if embeddings:
            stacked    = np.stack(embeddings, axis=0)
            mean_emb   = stacked.mean(axis=0)
            mean_emb  /= max(np.linalg.norm(mean_emb), 1e-8)
            identity_embeddings[identity_dir.name] = mean_emb.astype("float32")
            sample_counts[identity_dir.name]       = len(embeddings)

    if not identity_embeddings:
        raise RuntimeError("No valid face embeddings were generated.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    gallery = {
        "architecture":  "resnet34",
        "model_path":    str(args.model),
        "embeddings":    identity_embeddings,
        "sample_counts": sample_counts,
        "distance_metric": "cosine",
        "method": "classification",
    }
    with out_path.open("wb") as fh:
        pickle.dump(gallery, fh)

    print(f"Saved gallery → {out_path}")
    print(f"Registered identities: {len(identity_embeddings)}")
    for identity, count in sample_counts.items():
        print(f"  - {identity}: {count} image(s)")
    if skipped:
        print(f"Skipped (no face detected): {len(skipped)}")


def _load_face_image(image_path: Path, cropper) -> np.ndarray | None:
    frame = cv2.imread(str(image_path))
    if frame is None:
        return None

    if cropper is not None:
        result = cropper(frame)
        if "face_image" in result:
            return result["face_image"]

    resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)


if __name__ == "__main__":
    main()
