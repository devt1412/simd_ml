#!/bin/bash
set -e  # Exit on error

echo "----- Stage 1: Install build essentials and wheel -----"
pip install --upgrade pip
pip install wheel setuptools cython numpy==1.23.5 build

echo "----- Stage 2: Install build dependencies -----"
pip install wheel  # Ensure wheel is definitely installed
python -m pip install --upgrade pip setuptools wheel

echo "----- Stage 3: Install remaining requirements -----"
# Install memory-intensive packages first
pip install numpy==1.23.5 pandas==1.5.3 scipy --no-cache-dir
pip install -r requirements.txt --no-cache-dir

echo "----- Stage 4: Clean up build artifacts -----"
rm -rf build/ *.egg-info/
python -c "import gc; gc.collect()"

echo "----- Verification -----"
python -c "
import sys, numpy as np, pandas as pd, psutil;
print(f'Python {sys.version}\nNumPy {np.__version__}\nPandas {pd.__version__}');
print(f'Available memory: {psutil.virtual_memory().available / (1024*1024*1024):.2f} GB');
"