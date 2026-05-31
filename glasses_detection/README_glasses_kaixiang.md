# Kaixiang Glasses Detection

Independent innovative feature for the attendance facial-recognition system.

## Task

Input: cropped 224x224 RGB face image.

Output: `with_glasses` or `without_glasses` with confidence.

This module is independent from recognition, liveness, emotion, and UI modules.

## Models

- MobileNetV2 + binary head
- EfficientNetB0 + binary head

Both models use transfer learning with partial fine-tuning:

1. Load ImageNet backbone weights as initialization.
2. Replace the classifier with a new binary head.
3. Freeze the backbone and train the binary head.
4. Unfreeze the last feature blocks and fine-tune.

The model is not used as a fully pre-trained model without training.

## Loss and Metrics

- Loss: `BCEWithLogitsLoss`
- Class imbalance: positive class weighting
- Metrics: accuracy, precision, recall, F1, FPS

## Dataset

Expected structure:

```text
datasets/glasses/
  train/
    with_glasses/
    without_glasses/
  val/
    with_glasses/
    without_glasses/
  test/
    with_glasses/
    without_glasses/
```

Check dataset:

```bash
python -m glasses_detection.check_glasses_dataset_kaixiang
```

Smoke tests:

```bash
python -m glasses_detection.train_glasses_mobilenetv2_kaixiang --run-name smoke --head-epochs 1 --finetune-epochs 1 --batch-size 16 --max-train-per-class 500 --max-val-per-class 200 --progress-every 25
python -m glasses_detection.train_glasses_efficientnetb0_kaixiang --run-name smoke --head-epochs 1 --finetune-epochs 1 --batch-size 16 --max-train-per-class 500 --max-val-per-class 200 --progress-every 25
```

Final training:

```bash
python -m glasses_detection.train_glasses_mobilenetv2_kaixiang --run-name final --max-train-per-class 6000 --max-val-per-class 1500 --progress-every 100
python -m glasses_detection.train_glasses_efficientnetb0_kaixiang --run-name final --max-train-per-class 6000 --max-val-per-class 1500 --progress-every 100
```

Evaluation:

```bash
python -m glasses_detection.evaluate_glasses_kaixiang --checkpoint models/glasses_mobilenetv2_kaixiang_final_best.pth --split test
python -m glasses_detection.evaluate_glasses_kaixiang --checkpoint models/glasses_efficientnetb0_kaixiang_final_best.pth --split test
```

Model comparison:

```bash
python -m glasses_detection.compare_glasses_models_kaixiang --split test
```

Webcam demo:

```bash
python -m glasses_detection.test_glasses_integration_kaixiang --checkpoint models/glasses_efficientnetb0_kaixiang_final_best.pth
```
