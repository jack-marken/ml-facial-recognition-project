import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np

from face_verification.metric_learning.recognition_kaixiang import (
    DEFAULT_GALLERY_PATH,
    DEFAULT_MODEL_PATH,
    generate_embedding,
)
from face_verification.metric_learning.siamese_models_kaixiang import (
    load_siamese_checkpoint,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    print("Saving gallery...")
    parser = argparse.ArgumentParser(
        description="Build Kaixiang Siamese recognition gallery from face folders."
    )
    parser.add_argument("--db-path", default="datasets/faces_db")
    parser.add_argument("--checkpoint", default=str(DEFAULT_MODEL_PATH), type=Path)
    parser.add_argument("--output", default=str(DEFAULT_GALLERY_PATH))
    parser.add_argument("--skip-detection", action="store_true")
    args = parser.parse_args()

    db_path = Path(args.db_path)
    output_path = Path(args.output)
    if not db_path.exists():
        raise FileNotFoundError(f"Face database not found: {db_path}")

    model, checkpoint, _ = load_siamese_checkpoint(args.checkpoint)
    cropper = None
    if not args.skip_detection:
        from detection.detector import detect_and_crop_face

        cropper = detect_and_crop_face

    identity_embeddings = {}
    identity_sample_embeddings = {}
    sample_counts = {}
    skipped = []

    for identity_dir in sorted(path for path in db_path.iterdir() if path.is_dir()):
        embeddings = []
        for image_path in sorted(identity_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            face_image = load_face_image(image_path, cropper=cropper)
            if face_image is None:
                skipped.append(str(image_path))
                continue
            embeddings.append(generate_embedding(model, face_image))

        if embeddings:
            stacked = np.stack(embeddings, axis=0)
            mean_embedding = stacked.mean(axis=0)
            mean_embedding = mean_embedding / max(np.linalg.norm(mean_embedding), 1e-8)
            identity_embeddings[identity_dir.name] = mean_embedding.astype("float32")
            identity_sample_embeddings[identity_dir.name] = [
                embedding.astype("float32") for embedding in embeddings
            ]
            sample_counts[identity_dir.name] = len(embeddings)

    if not identity_embeddings:
        raise RuntimeError("No valid gallery embeddings were generated.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    gallery = {
        "model_name": checkpoint["model_name"],
        "model_path": str(args.checkpoint),
        "embeddings": identity_embeddings,
        "sample_embeddings": identity_sample_embeddings,
        "sample_counts": sample_counts,
        "distance_metric": "euclidean_on_l2_embeddings",
    }
    with output_path.open("wb") as file:
        pickle.dump(gallery, file)

    print(f"Saved gallery: {output_path}")
    print(f"Registered identities: {len(identity_embeddings)}")
    for identity, count in sample_counts.items():
        print(f"- {identity}: {count} image(s)")
    if skipped:
        print(f"Skipped images: {len(skipped)}")


def load_face_image(image_path: Path, cropper):
    frame = cv2.imread(str(image_path))
    if frame is None:
        return None

    if cropper is not None:
        result = cropper(frame)
        if "face_image" in result:
            return result["face_image"]

    resized_bgr = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)


if __name__ == "__main__":
    main()
