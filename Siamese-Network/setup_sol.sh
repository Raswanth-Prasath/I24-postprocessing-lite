#!/bin/bash

# Setup script for Siamese Network on Sol
# Run this first: bash setup_sol.sh

echo "========================================================================"
echo "SIAMESE NETWORK SETUP FOR SOL"
echo "========================================================================"
echo "Start Time: $(date)"
echo "Current Directory: $(pwd)"
echo "========================================================================"

# Create necessary directories
echo -e "\nStep 1: Creating directories..."
mkdir -p logs
mkdir -p outputs
echo "✓ Created logs/ directory"
echo "✓ Created outputs/ directory"

# Check Python version
echo -e "\nStep 2: Checking Python installation..."

# Try i24 conda environment first
PYTHON_BIN="/home/raswanth/.conda/envs/i24/bin/python"
if [ -f "$PYTHON_BIN" ]; then
    echo "✓ Found custom Python: $PYTHON_BIN"
    $PYTHON_BIN --version
else
    echo "Custom Python not found, checking system Python..."
    PYTHON_BIN="python"
    python --version
    if [ $? -eq 0 ]; then
        echo "✓ System Python found"
    else
        echo "✗ Python not found. Please install Python or update PYTHON_BIN path in scripts"
        exit 1
    fi
fi

# Install PyTorch and dependencies
echo -e "\nStep 3: Installing dependencies..."
echo "This may take a few minutes..."

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo "✗ requirements.txt not found!"
    exit 1
fi

# Install PyTorch with CUDA support
echo -e "\nInstalling PyTorch with CUDA support..."
$PYTHON_BIN -m pip install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
echo -e "\nInstalling other packages from requirements.txt..."
$PYTHON_BIN -m pip install --user -r requirements.txt

# Verify installation
echo -e "\nStep 4: Verifying installation..."
$PYTHON_BIN -c "
import torch
import numpy as np
import pandas as pd
import matplotlib
import sklearn
print('✓ All packages imported successfully')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'Number of GPUs: {torch.cuda.device_count()}')
"

if [ $? -eq 0 ]; then
    echo -e "\n========================================================================"
    echo "✓ SETUP COMPLETED SUCCESSFULLY!"
    echo "========================================================================"
    echo "End Time: $(date)"
    echo ""
    echo "Installation Summary:"
    echo "  ✓ Directories created (logs/, outputs/)"
    echo "  ✓ Python verified"
    echo "  ✓ PyTorch installed with CUDA support"
    echo "  ✓ All dependencies installed"
    echo ""
    echo "Next steps:"
    echo "  1. Test implementation:  sbatch run_test.sh"
    echo "  2. Start training:       sbatch run_train.sh"
    echo "  3. Monitor progress:     tail -f logs/train_*.out"
    echo "  4. Check job status:     squeue -u $USER"
    echo ""
    echo "For help, see: SOL_INSTRUCTIONS.md"
    echo "========================================================================"
else
    echo -e "\n========================================================================"
    echo "✗ SETUP FAILED"
    echo "========================================================================"
    echo "Please check error messages above."
    echo "Common issues:"
    echo "  - Python not found: Update PYTHON_BIN path in setup_sol.sh"
    echo "  - Package installation failed: Check internet connection"
    echo "  - CUDA not available: May need to load CUDA module"
    echo "========================================================================"
    exit 1
fi
