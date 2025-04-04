#!/bin/bash
# render-build.sh
echo "----- Installing dependencies with exact versions -----"
pip install -r requirements.txt --no-cache-dir

echo "----- Verifying installations -----"
pip freeze

echo "----- Checking numpy/pandas compatibility -----"
python -c "import numpy as np; import pandas as pd; print(f'numpy: {np.__version__}, pandas: {pd.__version__}')"