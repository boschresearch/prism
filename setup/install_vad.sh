#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Module Management ---
echo "Unloading potentially conflicting GCC and CUDA modules..."
module unload cuda  
module unload gcc

echo "Loading GCC 7.5 and CUDA 11.1 modules..."
module load gcc/7.5.0  
module load cuda/11.1

# --- Pip Installation ---
echo "Installing PyTorch, TorchVision (cu111)..."
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html

echo "Installing mmcv-full==1.4.0 (might take a while)..."
pip install --no-cache-dir mmcv-full==1.4.0

echo "Installing mmdet==2.14.0..."
pip install --no-cache-dir mmdet==2.14.0

echo "Installing mmsegmentation==0.14.1..."
pip install --no-cache-dir mmsegmentation==0.14.1

echo "Installing timm..."
pip install timm

# --- Build a simple package --- # 
echo "Build simple vad package"
pip install -e .

# --- MMDetection3D Setup ---
# Define the path to the mmdetection3d directory.  Make this an argument to the script for flexibility.
MMDET3D_PATH="${1:-../}" 
if [ ! -d "$MMDET3D_PATH/mmdetection3d" ]; then
  echo "Cloning mmdetection3d..."
  git clone https://github.com/open-mmlab/mmdetection3d.git
  mv mmdetection3d "$MMDET3D_PATH" # move the cloned directory to the desired location
  MMDET3D_PATH="$MMDET3D_PATH/mmdetection3d"
else
  echo "mmdetection3d directory already exists at $MMDET3D_PATH/mmdetection3d. Skipping clone."
  MMDET3D_PATH="$MMDET3D_PATH/mmdetection3d"
fi
cd "$MMDET3D_PATH"

echo "Checking out version v0.17.1..."
git checkout -f v0.17.1

echo "Installing mmdetection3d in develop mode..."
#python setup.py develop
pip install -e . --no-cache-dir

echo "Installing nuscenes-devkit==1.1.9..."
pip install nuscenes-devkit==1.1.9

echo "Installing additional dependencies..."
pip install similaritymeasures==0.7.0
pip install yapf==0.40.1
pip install tensorboard==2.9.0
pip install setuptools==59.5.0
pip install protobuf==3.20.3
pip install wandb==0.20.1

echo "VAD installation complete."