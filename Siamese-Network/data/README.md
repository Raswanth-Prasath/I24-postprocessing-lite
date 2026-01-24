# I24-3D Dataset

This folder contains the I-24 MOTION 3D dataset used for training and evaluation.

## Files

Required files for Siamese Network training:

- **GT_i.json** (24 MB) - Ground truth trajectories, scenario i (free-flow)
- **GT_ii.json** (13 MB) - Ground truth trajectories, scenario ii (snowy/slow)
- **GT_iii.json** (60 MB) - Ground truth trajectories, scenario iii (congested)
- **RAW_i.json** (42 MB) - Raw trajectory fragments, scenario i
- **RAW_ii.json** (19 MB) - Raw trajectory fragments, scenario ii
- **RAW_iii.json** (48 MB) - Raw trajectory fragments, scenario iii

**Total size:** ~203 MB

## Dataset Scenarios

### Scenario i (Free-Flow Traffic)
- Duration: 60 seconds
- Ground truth vehicles: 313
- Raw fragments: 789
- Conditions: Normal traffic flow

### Scenario ii (Snowy/Slow Traffic)
- Duration: 51 seconds
- Ground truth vehicles: 99
- Raw fragments: 411
- Conditions: Snow, reduced speeds

### Scenario iii (Congested Traffic)
- Duration: 50 seconds
- Ground truth vehicles: 266
- Raw fragments: 1,112
- Conditions: Stop-and-go traffic

## Data Format

Each JSON file contains an array of trajectory/fragment dictionaries with:

```json
{
  "timestamp": [array of timestamps],
  "x_position": [array of longitudinal positions in feet],
  "y_position": [array of lateral positions in feet],
  "velocity": [array of speeds in feet/second],
  "length": [array of vehicle lengths],
  "width": [array of vehicle widths],
  "height": [array of vehicle heights],
  "direction": 1 or -1 (eastbound/westbound),
  "_id": "unique fragment ID",
  "gt_ids": [[{"$oid": "ground truth vehicle ID"}]]
}
```

## Source

This data is from the I-24 MOTION project:
- Website: https://i24motion.org/
- Papers:
  - Wang et al. (2024) "Automatic vehicle trajectory data reconstruction at scale"
  - Wang et al. (2023) "Online Min Cost Circulation for Multi-Object Tracking on Fragments"

## Usage

The `siamese_dataset.py` automatically loads these files when training the Siamese network.

No manual data loading required - the dataset loader handles everything.

## For Sol

These files are already included in this folder for easy upload to Sol.

The code uses relative paths, so no path modifications needed!
