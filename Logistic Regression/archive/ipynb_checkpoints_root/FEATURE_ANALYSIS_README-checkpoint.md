# Feature Analysis and Selection for Logistic Regression

This guide helps you identify and fix three key issues in your logistic regression model:
1. **Multicollinearity** - Features that are highly correlated with each other
2. **Model Interpretability** - Understanding which features matter most
3. **Overfitting** - Model performs well on training data but poorly on new data

## Quick Start

### Step 1: Run Feature Analysis
```bash
cd "I24-postprocessing-lite/Logistic Regression"
python feature_analysis.py
```

This will analyze both the basic (28 features) and advanced (47 features) models and generate:
- VIF (Variance Inflation Factor) analysis
- Correlation matrices
- Feature importance rankings
- Learning curves to detect overfitting
- Regularization analysis
- Recommendations for which features to remove

**Output directory:** `feature_analysis_outputs/`

### Step 2: Apply Feature Selection
```bash
python feature_selection.py
```

This will automatically:
- Remove features with high multicollinearity (VIF > 10)
- Remove features with low importance (|coefficient| < 0.05)
- Remove one feature from each highly correlated pair (correlation > 0.85)
- Create optimized datasets
- Compare performance before/after

**Output files:**
- `training_dataset_combined_optimized.npz` - Optimized basic model
- `training_dataset_advanced_optimized.npz` - Optimized advanced model

**Output directory:** `feature_selection_outputs/`

### Step 3: Train with Optimized Features
```bash
python train_and_evaluate.py --use-optimized
```

Or modify `train_and_evaluate.py` to load the `*_optimized.npz` files instead.

---

## Understanding the Problems

### Problem 1: Multicollinearity

**What is it?**
When two or more features are highly correlated, they provide redundant information. This causes:
- Unstable coefficient estimates
- Difficulty interpreting feature importance
- Inflated standard errors
- Poor generalization

**How to detect:**
1. **VIF (Variance Inflation Factor)**
   - VIF < 5: No problem
   - VIF 5-10: Moderate multicollinearity
   - VIF > 10: **Severe multicollinearity - REMOVE FEATURE**

2. **Correlation Matrix**
   - |correlation| > 0.8: Features are highly correlated
   - Consider removing one from the pair

**Example from your data:**
```
Feature                            VIF       Status
velocity_fit_x                    45.23     HIGH (Remove)
velocity_actual_x_mean            42.15     HIGH (Remove)
projection_error_x_mean           12.34     HIGH (Remove)
```

These features are likely measuring the same underlying concept (x-direction velocity), so you should keep only the most important one.

### Problem 2: Model Interpretability

**What is it?**
In logistic regression, you want to understand which features drive predictions. Features with very small coefficients contribute little to the model.

**How to detect:**
Look at feature coefficients (weights):
- High |coefficient| (> 0.5): Very important
- Medium |coefficient| (0.1 - 0.5): Moderately important
- Low |coefficient| (< 0.1): **Minimal importance - CONSIDER REMOVING**

**Benefits of removing low-importance features:**
- Simpler model (easier to explain)
- Faster prediction
- Less prone to overfitting
- Better generalization

**Example:**
```
Feature                  Coefficient    Importance
time_gap                    1.234      High - KEEP
spatial_gap                 0.876      High - KEEP
curvature_b_std             0.034      Low - REMOVE
height_diff                 0.012      Low - REMOVE
```

### Problem 3: Overfitting

**What is it?**
Model learns the training data too well (including noise), performing poorly on new data.

**How to detect:**
1. **Learning Curves**
   - Training score >> Validation score → **Overfitting**
   - Both scores converge → Good generalization

2. **Train-Test Gap**
   - Gap < 0.05: Good
   - Gap 0.05-0.10: Moderate overfitting
   - Gap > 0.10: **Severe overfitting**

**Example:**
```
Training ROC-AUC:   0.9823
Test ROC-AUC:       0.8645
Gap:                0.1178  ← OVERFITTING!
```

**How to fix:**
1. **Remove features** (reduces model complexity)
2. **Increase regularization** (decrease C parameter)
3. **Use cross-validation** (already implemented)
4. **Get more training data**

---

## Interpreting the Analysis Results

### 1. VIF Analysis Output

```
VARIANCE INFLATION FACTOR (VIF) ANALYSIS
================================================================================
Feature                                        VIF       Status
velocity_fit_x                               45.234     HIGH (Remove)
velocity_actual_x_mean                       42.156     HIGH (Remove)
projection_error_x_mean                      12.456     HIGH (Remove)
spatial_gap                                   4.123     LOW (Good)
time_gap                                      2.345     LOW (Good)
```

**Action:** Remove all features with VIF > 10

### 2. Correlation Matrix

![Correlation Matrix](feature_analysis_outputs/correlation_matrix_advanced_model.png)

**Look for:**
- Dark red/blue cells (high correlation)
- Remove one feature from highly correlated pairs

### 3. Feature Importance

```
TOP 20 MOST IMPORTANT FEATURES
--------------------------------------------------------------------------------
Feature                         Coefficient    Abs_Coefficient
time_gap                           -1.2345          1.2345
spatial_gap                        -0.9876          0.9876
bhattacharyya_distance             -0.8765          0.8765
velocity_diff                       0.7654          0.7654
...
height_diff                         0.0234          0.0234  ← Low importance
curvature_b_std                     0.0123          0.0123  ← Low importance
```

**Action:** Consider removing features with |coefficient| < 0.05-0.1

### 4. Learning Curves

![Learning Curves](feature_analysis_outputs/learning_curves_advanced_model.png)

**Interpretation:**
- **Converging curves**: Good! Model generalizes well
- **Large persistent gap**: Overfitting - increase regularization or remove features
- **Both curves low**: Underfitting - add more features or reduce regularization

### 5. Regularization Analysis

```
Optimal C value: 0.3162
CV Score at optimal C: 0.9234
Test Score at optimal C: 0.9123

Current C = 1.0:
  CV Score: 0.9012
  Test Score: 0.8945

⚠ RECOMMENDATION: Use stronger regularization (C = 0.3162 instead of 1.0)
```

**Action:** Update your model training to use the optimal C value:
```python
model = LogisticRegression(
    class_weight='balanced',
    random_state=42,
    max_iter=1000,
    C=0.3162  # Use optimal value instead of 1.0
)
```

---

## Feature Selection Strategy

The `feature_selection.py` script uses a systematic 3-step approach:

### Step 1: Remove High VIF Features (Iterative)
- Compute VIF for all features
- Remove feature with highest VIF
- Recompute VIF (removing one feature changes others' VIF)
- Repeat until all VIF < 10

### Step 2: Remove Low Importance Features
- Train model on remaining features
- Remove features with |coefficient| < 0.05
- These contribute minimal predictive power

### Step 3: Remove Highly Correlated Features
- Find feature pairs with |correlation| > 0.85
- For each pair, keep the more important feature
- Remove the less important one

---

## Performance Comparison

After feature selection, you should see:

```
PERFORMANCE COMPARISON
================================================================================
Metric                         Original        Selected        Change
--------------------------------------------------------------------------------
ROC-AUC (Test)                0.9145          0.9167          +0.0022 (+0.24%)
Avg Precision (Test)          0.9234          0.9256          +0.0022 (+0.24%)
CV Score (Mean)               0.9012          0.9089          +0.0077 (+0.85%)

OVERFITTING CHECK:

Original Model:
  Train AUC: 0.9823
  Test AUC:  0.9145
  Gap:       0.0678

Feature-Selected Model:
  Train AUC: 0.9634
  Test AUC:  0.9167
  Gap:       0.0467

✓ Overfitting REDUCED by 0.0211 (better generalization)
```

**What to look for:**
- Test performance maintained or improved ✓
- Cross-validation score improved ✓
- Overfitting gap reduced ✓
- Fewer features (simpler model) ✓

---

## Expected Results

### Basic Model (28 features)
**Before feature selection:**
- Features: 28
- ROC-AUC: ~0.91
- Features with high VIF: 5-8
- Features with low importance: 8-12
- Expected to keep: ~15-20 features

**After feature selection:**
- Features: ~15-20
- ROC-AUC: ~0.91 (maintained or slightly improved)
- Overfitting gap reduced by ~0.02-0.04
- Model simpler and more interpretable

### Advanced Model (47 features)
**Before feature selection:**
- Features: 47
- ROC-AUC: ~0.92
- Features with high VIF: 10-15
- Features with low importance: 12-18
- Expected to keep: ~20-30 features

**After feature selection:**
- Features: ~20-30
- ROC-AUC: ~0.92 (maintained or slightly improved)
- Overfitting gap reduced by ~0.03-0.06
- Significant complexity reduction

---

## Files Generated

### feature_analysis.py outputs:
```
feature_analysis_outputs/
├── correlation_matrix_basic_model.png
├── correlation_matrix_advanced_model.png
├── feature_importance_basic_model.png
├── feature_importance_advanced_model.png
├── learning_curves_basic_model.png
├── learning_curves_advanced_model.png
├── regularization_analysis_basic_model.png
├── regularization_analysis_advanced_model.png
└── feature_removal_recommendations.txt
```

### feature_selection.py outputs:
```
feature_selection_outputs/
├── feature_selection_comparison_basic_model.png
├── feature_selection_comparison_advanced_model.png
├── removed_features_basic_model.txt
└── removed_features_advanced_model.txt

+ Root directory:
├── training_dataset_combined_optimized.npz (NEW)
└── training_dataset_advanced_optimized.npz (NEW)
```

---

## Troubleshooting

### Issue: "ValueError: Input contains NaN"
**Cause:** Some features have invalid values (NaN, inf)
**Solution:** Check your feature extraction code in `advanced_features.py`. Make sure error handling returns valid fallback values (not NaN).

### Issue: "All features removed!"
**Cause:** Thresholds too strict (VIF threshold too low, importance threshold too high)
**Solution:** Adjust thresholds in `feature_selection.py`:
- Increase VIF threshold from 10 to 15
- Decrease importance threshold from 0.05 to 0.01

### Issue: "Performance degrades after feature selection"
**Cause:** Removed important features
**Solution:**
- Use less aggressive thresholds
- Manually review removed features
- Use domain knowledge to keep critical features (e.g., time_gap, spatial_gap)

### Issue: "Still overfitting after feature selection"
**Cause:** Too many features remaining or need stronger regularization
**Solution:**
- Apply more aggressive feature selection
- Use smaller C value (e.g., C=0.1 instead of 1.0)
- Use L1 regularization (penalty='l1') for automatic feature selection

---

## Advanced Options

### Manual Feature Selection

If you want to manually control which features to keep:

```python
# In feature_selection.py, add this function:
def manual_feature_selection(X, feature_names, features_to_keep):
    """Keep only specified features"""
    keep_indices = [i for i, f in enumerate(feature_names) if f in features_to_keep]
    return X[:, keep_indices], [feature_names[i] for i in keep_indices]

# Example usage:
important_features = [
    'time_gap',
    'spatial_gap',
    'velocity_diff',
    'bhattacharyya_distance',
    'length_diff',
    # ... add more
]
X_manual, features_manual = manual_feature_selection(X, feature_names, important_features)
```

### Custom VIF Threshold

```python
# More aggressive (remove more features):
X_selected, features, removed = select_features_iterative_vif(
    X, y, feature_names, vif_threshold=5  # Instead of 10
)

# More conservative (keep more features):
X_selected, features, removed = select_features_iterative_vif(
    X, y, feature_names, vif_threshold=15  # Instead of 10
)
```

### Use L1 Regularization for Automatic Feature Selection

```python
from sklearn.linear_model import LogisticRegressionCV

# L1 regularization automatically pushes unimportant coefficients to zero
model = LogisticRegressionCV(
    penalty='l1',
    solver='saga',
    cv=5,
    class_weight='balanced',
    random_state=42
)
model.fit(X_scaled, y)

# Features with non-zero coefficients are selected
selected_features = [f for f, c in zip(feature_names, model.coef_[0]) if abs(c) > 1e-6]
```

---

## Next Steps

After running the feature analysis and selection:

1. **Review the results**
   - Check which features were removed and why
   - Verify that important domain knowledge features are retained
   - Review the performance comparison

2. **Update your training pipeline**
   - Use the `*_optimized.npz` datasets
   - Or incorporate the feature selection into your training script
   - Use the optimal C value from regularization analysis

3. **Integrate with the main pipeline**
   - Update `stitch_cost_interface.py` if using LR-based cost function
   - Retrain and save the optimized model
   - Test on the full I24 pipeline

4. **Monitor performance**
   - Run MOT evaluation on all three scenarios
   - Compare with baseline Bhattacharyya cost function
   - Check if the simpler model generalizes better

---

## Questions?

If you encounter issues or need help interpreting the results:
1. Check the generated visualization plots
2. Review the `removed_features_*.txt` files
3. Look at the feature importance rankings
4. Compare learning curves before/after

The goal is to create a simpler, more interpretable model that generalizes better while maintaining (or improving) predictive performance.
