# Logistic Regression Diagnostics Summary

- Dataset: `Logistic Regression/data/training_dataset_v5_diverse.npz`
- Model: `Logistic Regression/model_artifacts/consensus_top10_v5_diverse.pkl`
- Samples: 6300
- Full feature count: 47
- Model feature count: 10
- Influence method: `statsmodels`

## OOF Metrics

- ROC-AUC: 0.988418
- PR-AUC: 0.982399
- Log Loss: 0.125835
- Brier Score: 0.029643
- Precision: 0.953474
- Recall: 0.975873
- Specificity: 0.952381
- F1: 0.964543
- ECE: 0.020578

## Influence Summary

- Cook's D threshold (4/n): 0.000635
- Leverage threshold (2k/n): 0.003492
- |Standardized residual| threshold: 3.0
- Flagged influential points (top-k window): 250

## Top Shortcut-Risk Features (full dataset)

| Feature | abs(corr) | KS | Risk score |
|---|---:|---:|---:|
| projection_error_x_max | 0.6265 | 0.8797 | 0.7531 |
| width_diff | 0.4709 | 0.6416 | 0.5563 |
| height_diff | 0.3720 | 0.6337 | 0.5028 |
| length_diff | 0.3180 | 0.6454 | 0.4817 |
| bhattacharyya_coeff | 0.4119 | 0.5378 | 0.4748 |
| bhattacharyya_distance | 0.1769 | 0.5378 | 0.3573 |
| projection_error_y_max | 0.3728 | 0.3130 | 0.3429 |
| y_diff | 0.3478 | 0.2737 | 0.3107 |
| projection_error_x_mean | 0.0849 | 0.4971 | 0.2910 |
| time_gap | 0.1785 | 0.3159 | 0.2472 |

- Metrics JSON: `/home/raswanth/I24/I24-postprocessing-lite/Logistic Regression/feature_selection_outputs/lr_metrics_training_dataset_v5_diverse.json`
- Influence CSV: `/home/raswanth/I24/I24-postprocessing-lite/Logistic Regression/feature_selection_outputs/lr_influence_points_training_dataset_v5_diverse.csv`