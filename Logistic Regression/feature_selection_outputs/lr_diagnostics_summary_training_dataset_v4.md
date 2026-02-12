# Logistic Regression Diagnostics Summary

- Dataset: `Logistic Regression/training_dataset_v4.npz`
- Model: `Logistic Regression/model_artifacts/consensus_top10_full47.pkl`
- Samples: 8274
- Full feature count: 47
- Model feature count: 10
- Influence method: `manual`

## OOF Metrics

- ROC-AUC: 0.997971
- PR-AUC: 0.996662
- Log Loss: 0.047799
- Brier Score: 0.010611
- Precision: 0.975616
- Recall: 0.996132
- Specificity: 0.975103
- F1: 0.985767
- ECE: 0.019873

## Influence Summary

- Cook's D threshold (4/n): 0.000483
- Leverage threshold (2k/n): 0.002659
- |Standardized residual| threshold: 3.0
- Flagged influential points (top-k window): 250

## Top Shortcut-Risk Features (full dataset)

| Feature | abs(corr) | KS | Risk score |
|---|---:|---:|---:|
| projection_error_x_max | 0.6818 | 0.9251 | 0.8034 |
| width_diff | 0.5196 | 0.9021 | 0.7109 |
| height_diff | 0.4475 | 0.8774 | 0.6625 |
| length_diff | 0.3562 | 0.8898 | 0.6230 |
| bhattacharyya_coeff | 0.5142 | 0.6091 | 0.5617 |
| bhattacharyya_distance | 0.3625 | 0.6091 | 0.4858 |
| time_gap | 0.2885 | 0.4834 | 0.3860 |
| projection_error_x_mean | 0.0631 | 0.5313 | 0.2972 |
| projection_error_x_std | 0.2491 | 0.2538 | 0.2515 |
| velocity_mismatch_x | 0.2378 | 0.2110 | 0.2244 |

- Metrics JSON: `/home/raswanth/I24/I24-postprocessing-lite/Logistic Regression/feature_selection_outputs/lr_metrics_training_dataset_v4.json`
- Influence CSV: `/home/raswanth/I24/I24-postprocessing-lite/Logistic Regression/feature_selection_outputs/lr_influence_points_training_dataset_v4.csv`