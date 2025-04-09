#!/bin/bash
set -e  # Exit on error

echo "----- Stage 1: Install build essentials and wheel -----"
pip install --upgrade pip
pip install wheel setuptools cython numpy==1.23.5 build

echo "----- Stage 2: Install build dependencies -----"
pip install wheel  # Ensure wheel is definitely installed
python -m pip install --upgrade pip setuptools wheel

echo "----- Stage 3: Install remaining requirements -----"
pip install -r requirements.txt --no-cache-dir

echo "----- Verification -----"
python -c "
import sys, numpy as np, pandas as pd;
print(f'Python {sys.version}\nNumPy {np.__version__}\nPandas {pd.__version__}');
"