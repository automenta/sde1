#!/bin/bash
# A script to clean the project of temporary files and build artifacts.

echo "Cleaning project..."

# Remove build artifacts
rm -rf build/
rm -rf dist/
rm -rf .eggs/
find . -name "*.egg-info" -exec rm -rf {} +
find . -name "*.egg" -exec rm -f {} +

# Remove Python cache and compiled files
find . -name "__pycache__" -exec rm -rf {} +
find . -name "*.pyc" -exec rm -f {} +
find . -name "*.pyo" -exec rm -f {} +

# Remove log files
find . -name "*.log" -exec rm -f {} +

# Remove downloaded data and model checkpoints
rm -rf data_mnist/
rm -rf checkpoints/

echo "Done."
