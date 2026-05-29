# Kaixiang vs Zhongyu Model Comparison

Evaluation was run locally on the same dataset split for each module.

- Liveness split: `datasets/liveness/test`
- Recognition split: `datasets/recognition/test`

Best liveness model by AUC: **Kaixiang EfficientNetB0**.
Best recognition model by AUC: **Kaixiang Triplet ResNet18**.

## Liveness Results

| owner | model | accuracy | balanced_accuracy | auc | precision | recall | f1 | best_threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Zhongyu | DenseNet121 | 0.865 | 0.865 | 0.909575 | 0.876289 | 0.85 | 0.862944 | 0.554263 |
| Zhongyu | ResNet50V2 | 0.84 | 0.84 | 0.894325 | 0.846939 | 0.83 | 0.838384 | 0.673472 |
| Kaixiang | EfficientNetB0 | 0.9375 | 0.9375 | 0.984225 | 0.953368 | 0.92 | 0.936387 | 0.671468 |
| Kaixiang | MobileNetV2 | 0.86 | 0.86 | 0.943825 | 0.813043 | 0.935 | 0.869767 | 0.082906 |

## Recognition Verification Results

| owner | model | accuracy | balanced_accuracy | auc | precision | recall | f1 | best_threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Zhongyu | Triplet EfficientNet-B0 | 0.975057 | 0.520069 | 0.779491 | 0.555556 | 0.040984 | 0.076336 | 0.568232 |
| Zhongyu | Triplet ResNet34 | 0.975263 | 0.512189 | 0.879965 | 0.75 | 0.02459 | 0.047619 | 0.946444 |
| Kaixiang | Siamese MobileNetV2 | 0.974644 | 0.503887 | 0.828879 | 0.333333 | 0.008197 | 0.016 | -0.039831 |
| Kaixiang | Siamese ResNet18 | 0.975263 | 0.512189 | 0.852248 | 0.75 | 0.02459 | 0.047619 | -0.014684 |
| Kaixiang | Triplet MobileNetV2 | 0.975469 | 0.512295 | 0.839054 | 1.0 | 0.02459 | 0.048 | -0.003255 |
| Kaixiang | Triplet ResNet18 | 0.975881 | 0.560418 | 0.908753 | 0.6 | 0.122951 | 0.204082 | -0.002841 |
