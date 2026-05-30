"""Build a gallery of registered face embeddings from datasets/faces_db."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np

from .embedding_model_zhongyu import (
    DEFAULT_ARCHITECTURE,
    generate_embedding,
    load_embedding_model,
)
from .recognition_zhongyu import DEFAULT_MODEL_PATH


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build recognition gallery embeddings.")
    parser.add_argument("--db-path", default="datasets/faces_db")
    parser.add_argument("--output", default="models/recognition_gallery_zhongyu.pkl")
    parser.add_argument(
        "--architecture",
        default=DEFAULT_ARCHITECTURE,
        choices=["efficientnet_b0", "resnet34"],
    )
    parser.add_argument("--model", default=None, help="Optional trained .pth checkpoint.")
    parser.add_argument(
        "--skip-detection",
        action="store_true",
        help="Use direct resize instead of detection crop for registration images.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    db_path = Path(args.db_path)
    output_path = Path(args.output)
    model_path = _resolve_model_path(args.architecture, args.model)

    if not db_path.exists():
        raise FileNotFoundError(f"Face database not found: {db_path}")

    model = load_embedding_model(architecture=args.architecture, model_path=model_path)
    identity_embeddings: dict[str, np.ndarray] = {}
    sample_counts: dict[str, int] = {}
    skipped: list[str] = []

    cropper = None
    if not args.skip_detection:
        from detection.detector import detect_and_crop_face

        cropper = detect_and_crop_face

    for identity_dir in sorted(path for path in db_path.iterdir() if path.is_dir()):
        embeddings = []
        for image_path in sorted(identity_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            face_image = _load_face_image(image_path, cropper=cropper)
            if face_image is None:
                skipped.append(str(image_path))
                continue

            embeddings.append(generate_embedding(model, face_image))

        if embeddings:
            stacked = np.stack(embeddings, axis=0)
            mean_embedding = stacked.mean(axis=0)
            mean_embedding = mean_embedding / max(np.linalg.norm(mean_embedding), 1e-8)
            identity_embeddings[identity_dir.name] = mean_embedding.astype("float32")
            sample_counts[identity_dir.name] = len(embeddings)

    if not identity_embeddings:
        raise RuntimeError("No valid face embeddings were generated.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    gallery = {
        "architecture": args.architecture,
        "model_path": str(model_path) if model_path else None,
        "embeddings": identity_embeddings,
        "sample_counts": sample_counts,
        "distance_metric": "cosine",
    }
    with output_path.open("wb") as file:
        pickle.dump(gallery, file)

    print(f"Saved gallery: {output_path}")
    print(f"Registered identities: {len(identity_embeddings)}")
    for identity, count in sample_counts.items():
        print(f"- {identity}: {count} image(s)")
    if skipped:
        print(f"Skipped images: {len(skipped)}")


def _load_face_image(image_path: Path, cropper) -> np.ndarray | None:
    frame = cv2.imread(str(image_path))
    if frame is None:
        return None

    if cropper is not None:
        result = cropper(frame)
        if "face_image" in result:
            return result["face_image"]

    resized_bgr = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)


def _resolve_model_path(architecture: str, model_path: str | None) -> str | None:
    if model_path:
        return model_path
    if architecture == DEFAULT_ARCHITECTURE and DEFAULT_MODEL_PATH is not None:
        return str(DEFAULT_MODEL_PATH)
    return None


if __name__ == "__main__":
    main()
