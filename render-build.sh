#!/bin/bash
set -e  # Exit on error

echo "----- Stage 1: Install build essentials -----"
pip install --upgrade pip
pip install wheel setuptools cython numpy==1.23.5

echo "----- Stage 2: Install remaining requirements -----"
pip install -r requirements.txt --no-cache-dir

echo "----- Verification -----"
python -c "
import sys, numpy as np, pandas as pd;
print(f'Python {sys.version}\nNumPy {np.__version__}\nPandas {pd.__version__}');
"