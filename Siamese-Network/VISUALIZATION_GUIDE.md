# Visualization Guide - Comparing Reconstruction Results

## Overview

The `visualize_results.py` script creates comprehensive time-space diagrams comparing:

1. **Raw Fragments** - Fragmented trajectories before reconstruction
2. **Ground Truth** - Complete vehicle trajectories (reference)
3. **Model Predictions** - Reconstructed trajectories using your trained model

## Visual Output

The script generates a **6-row comparison plot**:

```
Row 1-2: Raw Fragments (EB and WB)
  └─ Shows fragmented trajectories with gaps

Row 3-4: Ground Truth (EB and WB)
  └─ Shows complete, correct trajectories

Row 5-6: Model Predictions (EB and WB)
  └─ Shows your model's reconstruction
```

Each column represents a different lane (Lane 1, Lane 2, etc.)

---

## Quick Start

### Option 1: Visualize Logistic Regression Results

```bash
# 1. Export LR model
cd "D:/ASU Academics/Thesis & Research/02_Code/Logistic-Regression"
python export_model_for_comparison.py

# 2. Run visualization
cd "../Siamese-Network"
python visualize_results.py
```

### Option 2: Visualize Siamese Network Results

```bash
# 1. Train Siamese model first (on Sol or locally)
cd "D:/ASU Academics/Thesis & Research/02_Code/Siamese-Network"
python train_siamese.py  # or submit to Sol

# 2. Run visualization
python visualize_results.py
```

---

## Configuration

Edit `visualize_results.py` (near line 535) to change settings:

```python
# Choose scenario to visualize
scenario = 'i'  # Options: 'i', 'ii', 'iii'

# Choose model type
model_type = 'lr'  # Options: 'lr' or 'siamese'
```

**Scenarios:**
- **'i'** - Free-flow traffic (60 sec, 313 vehicles, 789 fragments)
- **'ii'** - Snowy/slow traffic (51 sec, 99 vehicles, 411 fragments)
- **'iii'** - Congested traffic (50 sec, 266 vehicles, 1,112 fragments)

---

## Step-by-Step Usage

### For Logistic Regression Comparison

**Step 1: Export your trained LR model**

```bash
cd "02_Code/Logistic-Regression"
python export_model_for_comparison.py
```

**Expected output:**
```
✓ Model saved to: outputs/lr_model.pkl
✓ Scaler saved to: outputs/scaler.pkl
✓ Feature names saved to: outputs/feature_names.pkl
```

**Step 2: Run visualization**

```bash
cd "../Siamese-Network"
python visualize_results.py
```

**Expected output:**
- Console statistics about fragments and trajectories
- Interactive plot window showing 3-way comparison
- Saved PNG file in `outputs/comparison_scenario_i.png`

---

### For Siamese Network Comparison

**Step 1: Train model** (if not done yet)

```bash
cd "02_Code/Siamese-Network"

# On Sol
sbatch run_train.sh

# Or locally (if you have GPU)
python train_siamese.py
```

**Step 2: Edit visualize_results.py**

Change line ~537 to:
```python
model_type = 'siamese'  # Use Siamese instead of LR
```

**Step 3: Run visualization**

```bash
python visualize_results.py
```

---

## Understanding the Output

### Statistics Printed

```
COMPARISON STATISTICS
Raw fragments: 789        ← Number of broken fragments
Ground truth vehicles: 313 ← Number of complete vehicles
Predicted trajectories: 250 ← Your model's reconstruction

Raw: EB=450, WB=339
GT:  EB=180, WB=133
```

**What to look for:**
- **Perfect reconstruction**: Predicted ≈ Ground truth count
- **Under-reconstruction**: Predicted < GT (fragments not linked)
- **Over-reconstruction**: Predicted > GT (wrong linkages)

### Visual Interpretation

**Good reconstruction:**
- ✅ Row 5-6 looks similar to Row 3-4 (GT)
- ✅ Continuous trajectories, no gaps
- ✅ Correct lane assignments

**Poor reconstruction:**
- ❌ Still shows fragmentation (like Row 1-2)
- ❌ Trajectories jump between lanes
- ❌ Fewer/more trajectories than GT

---

## Customization

### Change Scenario

```python
# In visualize_results.py, line ~536
scenario = 'iii'  # Try congested traffic
```

### Adjust Reconstruction Threshold

```python
# In visualize_results.py, line ~288 (for LR)
threshold=0.5  # Lower = more aggressive linking
               # Higher = more conservative
```

**Try different thresholds:**
- `0.3` - More links (may create false positives)
- `0.5` - Balanced (default)
- `0.7` - Fewer links (more conservative)

### Change Plot Style

```python
# In plot_comparison function, adjust:
linewidth=2.5    # Thicker lines
alpha=0.7        # Transparency
markersize=1     # Point size for fragments
```

---

## Output Files

All outputs saved to `outputs/`:

```
outputs/
├── comparison_scenario_i.png    # Scenario i comparison
├── comparison_scenario_ii.png   # Scenario ii comparison
├── comparison_scenario_iii.png  # Scenario iii comparison
└── best_accuracy.pth            # Trained Siamese model
```

**Plot specifications:**
- Format: PNG
- Resolution: 300 DPI (publication quality)
- Size: Large (depends on number of lanes)

---

## Troubleshooting

### Issue: "LR model not found"

```
Warning: LR model not found at outputs/lr_model.pkl
```

**Solution:**
```bash
cd "02_Code/Logistic-Regression"
python export_model_for_comparison.py
```

### Issue: "Siamese model not found"

```
Warning: Siamese model not found at outputs/best_accuracy.pth
```

**Solution:**
```bash
# Train the model first
python train_siamese.py

# Or wait for Sol job to complete
```

### Issue: "Dataset not found"

```
FileNotFoundError: data/GT_i.json not found
```

**Solution:**
- Make sure you're in the `Siamese-Network` folder
- Verify `data/` folder contains 6 JSON files
- See `START_HERE.md` for setup instructions

### Issue: "No trajectories reconstructed"

```
Predicted trajectories: 0
```

**Possible causes:**
1. Threshold too high (try lowering to 0.3)
2. Model not trained properly
3. Feature extraction mismatch

**Solution:**
```python
# Lower threshold
threshold=0.3  # In reconstruct_trajectories_lr function
```

---

## Comparing Models

### Compare LR vs Siamese

**Method 1: Run both and compare visually**

```bash
# 1. Visualize LR
python visualize_results.py  # model_type='lr'

# 2. Change to Siamese
# Edit visualize_results.py: model_type='siamese'
python visualize_results.py

# 3. Compare the two output PNGs
```

**Method 2: Side-by-side code**

Create a script that generates both in one plot (requires modification of `visualize_results.py`).

---

## For Your Thesis

### Include in Results Chapter

**Figure caption example:**

> **Figure X:** Time-space diagram comparison for scenario i (free-flow traffic).
> Top: Raw trajectory fragments (789 fragments). Middle: Ground truth complete
> trajectories (313 vehicles). Bottom: Reconstructed trajectories using [Model Name]
> ([Predicted Count] trajectories). Eastbound (EB) and Westbound (WB) lanes shown
> separately. Colors represent individual vehicles/trajectories.

### Metrics to Report

From the console output:
```python
fragments_to_vehicles_ratio = raw_fragments / gt_vehicles
reconstruction_accuracy = predicted / gt_vehicles * 100
```

**Example:**
- Raw: 789 fragments from 313 vehicles (2.52 fragments/vehicle)
- Predicted: 305 trajectories
- Reconstruction rate: 97.4% (305/313)

---

## Quick Reference Commands

```bash
# Export LR model
cd Logistic-Regression
python export_model_for_comparison.py

# Visualize scenario i with LR
cd ../Siamese-Network
python visualize_results.py

# Visualize scenario ii with Siamese
# (Edit: scenario='ii', model_type='siamese')
python visualize_results.py

# Visualize all scenarios
for scenario in i ii iii; do
    # Edit scenario in script, then:
    python visualize_results.py
done
```

---

## Advanced: Batch Processing

Create a script to visualize all scenarios:

```python
# visualize_all.py
scenarios = ['i', 'ii', 'iii']
for scenario in scenarios:
    print(f"Processing scenario {scenario}...")
    # Run visualization with scenario
    # ... (modify visualize_results.py to accept CLI args)
```

---

## Questions?

**Common questions:**

Q: Can I compare Bhattacharyya baseline?
A: Yes, integrate the baseline cost function similar to LR model.

Q: Can I visualize on Sol?
A: Yes, but you need X11 forwarding or save plots without displaying:
```python
# Add before plt.show():
plt.savefig(output_path)
# Comment out:
# plt.show()
```

Q: How do I automate comparison?
A: Modify the script to calculate and print MOT metrics (MOTA, MOTP, etc.)

---

**Status:** Ready to use after exporting LR model or training Siamese model ✅
