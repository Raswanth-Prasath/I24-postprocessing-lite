# Siamese Network - File Checklist for Sol

## Pre-Upload Verification

Before uploading to Sol, verify you have all these files:

### ✅ Core Python Files (Required)
- [ ] `siamese_dataset.py` - Dataset loader for trajectory pairs
- [ ] `siamese_model.py` - Siamese network architecture
- [ ] `train_siamese.py` - Training script
- [ ] `evaluate_siamese.py` - Evaluation script
- [ ] `pipeline_integration.py` - Integration with I24 pipeline

### ✅ Configuration Files (Required)
- [ ] `requirements.txt` - Python dependencies

### ✅ Shell Scripts (Required for Sol)
- [ ] `setup_sol.sh` - Setup and installation script
- [ ] `run_test.sh` - Testing SLURM script
- [ ] `run_train.sh` - Training SLURM script

### ✅ Documentation (Recommended)
- [ ] `README.md` - General usage guide
- [ ] `IMPLEMENTATION_SUMMARY.md` - Technical overview
- [ ] `SOL_INSTRUCTIONS.md` - Sol-specific instructions
- [ ] `FILE_CHECKLIST.md` - This file

### ✅ Data Access (Critical)
- [ ] I24-3D dataset accessible on Sol
  - [ ] `GT_i.json`
  - [ ] `GT_ii.json`
  - [ ] `GT_iii.json`
  - [ ] `RAW_i.json`
  - [ ] `RAW_ii.json`
  - [ ] `RAW_iii.json`

**Dataset path to verify/update in `siamese_dataset.py` (line 21):**
```python
self.dataset_dir = Path("YOUR_SOL_PATH/I24-3D")
```

---

## Quick Verification Script

Run this on Sol after upload:

```bash
cd path/to/Siamese-Network

echo "Checking files..."
for file in siamese_dataset.py siamese_model.py train_siamese.py evaluate_siamese.py pipeline_integration.py requirements.txt setup_sol.sh run_test.sh run_train.sh; do
    if [ -f "$file" ]; then
        echo "✓ $file"
    else
        echo "✗ $file - MISSING!"
    fi
done

echo ""
echo "Checking dataset..."
# Update this path to your actual dataset location
DATASET_DIR="YOUR_PATH/I24-3D"
for file in GT_i.json GT_ii.json GT_iii.json RAW_i.json RAW_ii.json RAW_iii.json; do
    if [ -f "$DATASET_DIR/$file" ]; then
        echo "✓ $file"
    else
        echo "✗ $file - MISSING!"
    fi
done
```

---

## Directory Structure on Sol

Your final structure should look like:

```
Siamese-Network/
├── siamese_dataset.py
├── siamese_model.py
├── train_siamese.py
├── evaluate_siamese.py
├── pipeline_integration.py
├── requirements.txt
├── setup_sol.sh
├── run_test.sh
├── run_train.sh
├── README.md
├── IMPLEMENTATION_SUMMARY.md
├── SOL_INSTRUCTIONS.md
├── FILE_CHECKLIST.md
├── logs/                    (created by setup)
└── outputs/                 (created by setup)
```

---

## File Sizes (for verification)

Approximate sizes to verify correct upload:

```
siamese_dataset.py          ~15 KB
siamese_model.py            ~14 KB
train_siamese.py            ~12 KB
evaluate_siamese.py         ~10 KB
pipeline_integration.py     ~10 KB
requirements.txt            ~1 KB
setup_sol.sh                ~2 KB
run_test.sh                 ~4 KB
run_train.sh                ~2 KB
README.md                   ~8 KB
IMPLEMENTATION_SUMMARY.md   ~15 KB
SOL_INSTRUCTIONS.md         ~12 KB
```

---

## After Upload to Sol

1. **Make scripts executable:**
```bash
chmod +x setup_sol.sh run_test.sh run_train.sh
```

2. **Update dataset path:**
```bash
nano siamese_dataset.py
# Change line 21 to your Sol path
```

3. **Run verification:**
```bash
bash setup_sol.sh
```

4. **Check Python imports:**
```bash
python -c "
import siamese_dataset
import siamese_model
print('✓ All imports successful')
"
```

---

## Critical Path Updates Needed

Before running on Sol, you MUST update this line in `siamese_dataset.py`:

**Line 21 - Current (Windows path):**
```python
self.dataset_dir = Path(r"D:\ASU Academics\Thesis & Research\01_Papers\Datasets\I24-3D")
```

**Update to (Sol path):**
```python
self.dataset_dir = Path("/home/YOUR_USERNAME/path/to/I24-3D")
```

Similarly in `train_siamese.py` and `evaluate_siamese.py`, update any hardcoded paths.

---

## Ready to Go Checklist

Before submitting first job:

- [ ] All Python files uploaded
- [ ] All shell scripts uploaded and executable
- [ ] Dataset path updated in `siamese_dataset.py`
- [ ] Output path updated in `train_siamese.py` (line 232)
- [ ] `setup_sol.sh` completed successfully
- [ ] `run_test.sh` passed all tests
- [ ] Ready to submit `run_train.sh`

---

## Minimal File Set

If you only want essential files:

**Absolutely Required:**
1. `siamese_dataset.py`
2. `siamese_model.py`
3. `train_siamese.py`
4. `requirements.txt`
5. `run_train.sh`

**Highly Recommended:**
6. `setup_sol.sh`
7. `run_test.sh`
8. `SOL_INSTRUCTIONS.md`

---

## Final Verification Command

```bash
# Run this on Sol to verify everything
cd path/to/Siamese-Network
python -c "
import os
from pathlib import Path

required_files = [
    'siamese_dataset.py',
    'siamese_model.py',
    'train_siamese.py',
    'requirements.txt'
]

print('Checking required files:')
all_present = True
for f in required_files:
    if Path(f).exists():
        print(f'✓ {f}')
    else:
        print(f'✗ {f} MISSING!')
        all_present = False

if all_present:
    print('\n✓ All required files present!')
    print('Ready to run: bash setup_sol.sh')
else:
    print('\n✗ Some files missing. Check above.')
"
```

---

**Status: All files created and ready for Sol deployment** ✅
