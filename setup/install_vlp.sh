#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

cd vlp || { echo "Directory 'vlp' not found!"; exit 1; }

# --- Install Dependencies ---
echo "Installing requirements from requirements.txt..."
pip install -r requirements.txt

echo "Installing vad_vlp in editable mode..."
pip install -e .

# --- Fix version mismatch --- #    
echo "Installing specific package versions to resolve potential conflicts of VLP ..."
pip install numpy==1.19.5
pip install path 
# test if those below needed 
pip install wget
pip install --force-reinstall setuptools==59.5.0
pip install more_itertools==10.5.0
echo "vad_vlp setup complete!"