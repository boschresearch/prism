# Procedure to install VAD (+ VLP) 

## Clone VAD
```bash
git clone --recursive https://github.com/boschresearch/prism.git
```
Make sure you also cloned `vlp` repo correctly. 

## Install Environment 

### Conda Env 
- Create and activate the environment
```bash 
conda create -n vad_vlp python=3.8 -y
conda activate vad_vlp
```
- Install missing libraries. This is done to solve `missing libmpfr.so.4 library` error. 
```bash
conda install -c conda-forge mpfr=3.1.4
```
- Explicitly add path to you env's `lib` folder to `LD_LIBRARY_PATH` (modify path accordingly)
```bash
export LD_LIBRARY_PATH=/home/hao1rng/.conda/envs/vad_vlp/lib:$LD_LIBRARY_PATH
```

### Install **VAD**
- Notes: entirely based on original [VAD guide](../docs/install.md).
- Execute install script:
```bash
find ./setup/ -name "*.sh" -exec chmod +x {} \;
PYTHONUNBUFFERED=1 ./setup/install_vad.sh
``` 

### Install VLP
- If you plan to use **VLP** models as well, run: 
```bash
PYTHONUNBUFFERED=1 ./setup/install_vlp.sh
```

## Prepare Data 
- Create symlink for nuscenes 
```bash 
cd Our-VAD
mkdir -p data/nuscenes
ln -s /fs/scratch/datasets/cr_dlp_motional_permissive/nuscenes/* ./data/nuscenes/
```

- Donwload custom annotations ([train](https://drive.google.com/file/d/1OVd6Rw2wYjT_ylihCixzF6_olrAQsctx/view?usp=sharing) file and [val](https://drive.google.com/file/d/16DZeA-iepMCaeyi57XSXL3vYyhrOQI9S/view?usp=sharing) file from google drive) and place them under `data/nuscenes_vad_ann`. 
> Unhortunately, I was not able to setup download via CLI, so you have to do it manually.

```bash 
cd Our-VAD
mkdir -p data/nuscenes_vad_ann
cd data/nuscenes_vad_ann
```

## Ckpts
```bash
cd Our-VAD
mkdir ckpts
cd ckpts 
wget https://download.pytorch.org/models/resnet50-19c8e357.pth
```
