# Kaixiang vs Zhongyu Recomparison

This report recomputes all models on the same local dataset splits and generates ROC, AUC, threshold, and score-distribution visuals.

Best liveness model by ROC-AUC: **kaixiang_efficientnetb0** (0.9724).
Best recognition model by ROC-AUC: **kaixiang_triplet_resnet18** (0.8880).

## Output Figures

- `recognition_model_auc_accuracy_fps.png`
- `recognition_roc_curves.png`
- `recognition_best_model_distance_metric_comparison.png`
- `recognition_best_model_score_distribution.png`
- `liveness_model_auc_accuracy_fps.png`
- `liveness_roc_curves.png`
- `liveness_best_model_score_distribution.png`
