# PRISM

This repository contains the implementation of "PRISM: Privileged Probabilistic Latent Supervision for End-to-End Autonomous Driving Motion Planning", authored by Volodymyr Havrylov, Faris Janjoš, Andreas Look, Jürgen Mathes, and Andreas Geiger, accepted at IROS26. It based on the VAD (Vectorized Scene Representation for Efficient Autonomous Driving) codebase.

## Setup 
- Follow [README](setup/README.md)

## Train 
- If you did not install VLP part of environment and run only VAD training, comment this [line](./projects/mmdet3d_plugin/__init__.py#L13)
- Prepare log dir
```bash 
mkdir logs 
```

- Modify desired parameters in `scripts/train_launcher.sh`, e.g.: queue, batch size, num of gpus, etc.
- Submit job 
```bash 
module load conda gcc/7.5.0 cuda/11.1
./scripts/train_launcher.sh
```

## Eval 
- Usually takes around 30 min
- If you did not install VLP part of environment and run only VAD training, comment this [line](./projects/mmdet3d_plugin/__init__.py#L13) 
- Prepare log dir
```bash 
mkdir -p logs/eval 
```
- Modify desired parameters in `scripts/eval_launcher.sh`, e.g.: config path, epoch to eval, etc.
- Submit job 
```bash 
module load conda gcc/7.5.0 cuda/11.1
./scripts/eval_launcher.sh
```
- Note: it creates certain folder structure that is assumed in other tools. 

## Other Actions

- `plot_l2_col_rate.py` produces metric plots (relies on evaluation folder structure created by eval script)
- `log_exp_to_wandb.py` logs evaluated runs to WandB (relies on evaluation folder structure created by eval script)
- Command to sync folder from remote to host: 
`rsync -avh --dry-run --info=progress2 -z   hao1rng@rng-dl01-login2.de.bosch.com:/home/hao1rng/core/VAD/exps/plots   ./thesis_plots/`
