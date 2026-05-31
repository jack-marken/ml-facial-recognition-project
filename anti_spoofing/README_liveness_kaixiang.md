# Kaixiang Liveness Detection

This folder contains Kaixiang's liveness detection implementation only.

The final liveness models must be trained or fine-tuned on the project dataset.
The scripts may initialize the backbone with ImageNet weights for transfer
learning, but they replace the original classifier with a custom binary head,
train the new head, then unfreeze final backbone blocks for fine-tuning.

## Dataset

Expected local-only structure:

```text
datasets/liveness/
  train/
    real/
    spoof/
  val/
    real/
    spoof/
  test/
    real/
    spoof/
```

Check the dataset:

```bash
python -m anti_spoofing.check_liveness_dataset_kaixiang
```

## Train

MobileNetV2 + binary head:

```bash
python -m anti_spoofing.train_liveness_mobilenetv2_kaixiang
```

EfficientNetB0 + binary head:

```bash
python -m anti_spoofing.train_liveness_efficientnetb0_kaixiang
```

The two entry scripts contain their own binary heads, freeze/unfreeze strategy,
and default tuning settings:

```text
MobileNetV2:
  batch_size=16
  head_epochs=5
  finetune_epochs=15
  unfreeze_blocks=6
  head_lr=1e-3
  finetune_lr=1e-4
  weight_decay=1e-4
  early_stopping_patience=5

EfficientNetB0:
  batch_size=8
  head_epochs=5
  finetune_epochs=15
  unfreeze_blocks=4
  head_lr=1e-3
  finetune_lr=5e-5
  weight_decay=1e-4
  early_stopping_patience=5
```

Use `--run-name` to keep tuning experiments instead of overwriting them:

```bash
python -m anti_spoofing.train_liveness_mobilenetv2_kaixiang --run-name exp_a --batch-size 8 --finetune-lr 5e-5
python -m anti_spoofing.train_liveness_efficientnetb0_kaixiang --run-name exp_a --unfreeze-blocks 2
```

Outputs are saved under `models/` by default:

```text
models/liveness_mobilenetv2_kaixiang_best.pth
models/liveness_efficientnetb0_kaixiang_best.pth
models/liveness_mobilenetv2_kaixiang_exp_a_best.pth
```

Do not commit datasets or model weights.

## Evaluate

```bash
python -m anti_spoofing.evaluate_liveness_kaixiang --checkpoint models/liveness_mobilenetv2_kaixiang_best.pth --split test
```

Reported metrics include accuracy, precision, recall, F1, confusion matrix,
optional ROC/AUC if scikit-learn is installed, and FPS.

## Integration Interface

Use the detection module to crop the face first:

```python
from detection.detector import detect_and_crop_face
from anti_spoofing.liveness_kaixiang import predict_liveness

detection_result = detect_and_crop_face(frame)
if "status" not in detection_result:
    liveness_result = predict_liveness(detection_result["face_image"])
```

Expected output:

```python
{"liveness": "REAL", "confidence": 0.93, "model": "mobilenetv2"}
```
