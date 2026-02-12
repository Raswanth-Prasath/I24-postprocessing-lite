# Logistic Regression Diagnostics Summary

- Dataset: `Logistic Regression/training_dataset_advanced.npz`
- Model: `Logistic Regression/model_artifacts/consensus_top10_full47.pkl`
- Samples: 2100
- Full feature count: 47
- Model feature count: 10
- Influence method: `manual`

## OOF Metrics

- ROC-AUC: 0.999364
- PR-AUC: 0.999302
- Log Loss: 0.037368
- Brier Score: 0.008885
- Precision: 0.981238
- Recall: 0.996190
- Specificity: 0.980952
- F1: 0.988658
- ECE: 0.017651

## Influence Summary

- Cook's D threshold (4/n): 0.001905
- Leverage threshold (2k/n): 0.010476
- |Standardized residual| threshold: 3.0
- Flagged influential points (top-k window): 177

## Top Shortcut-Risk Features (full dataset)

| Feature | abs(corr) | KS | Risk score |
|---|---:|---:|---:|
| projection_error_x_max | 0.6977 | 0.7886 | 0.7432 |
| y_diff | 0.6840 | 0.7752 | 0.7296 |
| projection_error_y_max | 0.6766 | 0.7581 | 0.7174 |
| bhattacharyya_coeff | 0.4310 | 0.7562 | 0.5936 |
| bhattacharyya_distance | 0.3127 | 0.7562 | 0.5344 |
| width_diff | 0.4123 | 0.4181 | 0.4152 |
| length_diff | 0.3324 | 0.4752 | 0.4038 |
| projection_error_x_mean | 0.1612 | 0.4933 | 0.3273 |
| spatial_gap | 0.1713 | 0.4505 | 0.3109 |
| duration_a | 0.3181 | 0.2695 | 0.2938 |

- Metrics JSON: `/home/raswanth/I24/I24-postprocessing-lite/Logistic Regression/feature_selection_outputs/lr_metrics_training_dataset_advanced.json`
- Influence CSV: `/home/raswanth/I24/I24-postprocessing-lite/Logistic Regression/feature_selection_outputs/lr_influence_points_training_dataset_advanced.csv`