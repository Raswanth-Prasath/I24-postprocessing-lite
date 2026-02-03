#!/bin/bash
# Setup script for pre-commit hooks

set -e

echo "Setting up pre-commit hooks for I24-postprocessing-lite..."

# Check if pre-commit is installed
if ! command -v pre-commit &> /dev/null; then
    echo "Installing pre-commit..."
    pip install pre-commit
fi

# Install the git hook scripts
echo "Installing pre-commit hooks..."
pre-commit install

# Optional: Run against all files for first-time setup
read -p "Do you want to run pre-commit against all files now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Running pre-commit on all files (this may take a while)..."
    pre-commit run --all-files || echo "Some checks failed. Fix them and run 'git add' again."
fi

echo ""
echo "✓ Pre-commit hooks installed successfully!"
echo ""
echo "Usage:"
echo "  - Hooks will run automatically on 'git commit'"
echo "  - To skip hooks: SKIP=hook-id git commit -m 'message'"
echo "  - To run manually: pre-commit run --all-files"
echo "  - To update hooks: pre-commit autoupdate"
echo ""
