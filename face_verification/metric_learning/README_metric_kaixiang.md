# Kaixiang Metric Learning Recognition

This folder contains Kaixiang's metric-learning recognition implementation.
It is separate from Zhongyu's triplet-learning files and uses Kaixiang-specific
filenames to avoid merge conflicts.

## Models

Two Siamese models are prepared for comparison:

```text
Siamese + ResNet18
Siamese + MobileNetV2
```

Both use ImageNet pretrained backbones only as transfer-learning
initialization. The original classification heads are removed and replaced with
projection heads that output L2-normalized face embeddings. Training uses
contrastive loss with labels:

```text
same identity      = 1
different identity = 0
```

The training sampler generates balanced positive and negative face pairs. Train
time augmentation is enabled by default to improve webcam/domain robustness:
horizontal flip, random crop/scale, small rotation, and brightness/contrast
jitter. Validation and test pairs remain deterministic and unaugmented for fair
ROC/AUC reporting.

## Dataset

Expected local-only structure:

```text
datasets/recognition/
  train/
    person_a/
    person_b/
  val/
    person_a/
    person_b/
  test/
    person_a/
    person_b/
```

Each identity folder should contain at least two images where possible, because
positive pairs require two images from the same identity.

Check the dataset:

```bash
python -m face_verification.metric_learning.check_metric_dataset_kaixiang
```

## Train

ResNet18:

```bash
python -m face_verification.metric_learning.train_siamese_resnet18_kaixiang --run-name final1
```

MobileNetV2:

```bash
python -m face_verification.metric_learning.train_siamese_mobilenetv2_kaixiang --run-name final1
```

The default tuned settings use 3000 sampled pairs per epoch. For a quick smoke
test, override the pair count and epoch count from the command line.

## Evaluate

```bash
python -m face_verification.metric_learning.evaluate_metric_kaixiang --checkpoint models/recognition_siamese_resnet18_kaixiang_final1_best.pth --split test
```

Evaluation reports:

```text
Euclidean ROC/AUC
Euclidean accuracy
best Euclidean distance threshold
cosine ROC/AUC
cosine accuracy
best cosine similarity threshold
inference FPS
```
