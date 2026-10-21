#!/bin/bash

# Script to launch training in an interactive job session. Used for debugging purposes. 

# ==============================================================================
# Training Configuration
# ==============================================================================

CONFIG_PATH=/home/hao1rng/core/VAD/vlp/vlp/configs/vad-fixed/probabilistic/slp_egoagent/multivar_normal_mlp/indep_var_mu_baseline/from_query/heavy/w1e-1_freeze_post2prior_grad/VAD_tiny_e2e_slp-egoagent_multivar-normal-mlp_indep-var-mu-baseline_from_Q_freeze_post2prior_grad_heavy_w1e-1.py

WORK_DIR="./exps/tmp/$(basename "$CONFIG_PATH" .py)_interactive"

NGPUS=1
MASTER_PORT=2333
LAUNCHER="pytorch"
SAMPLES_PER_GPU=2
DETERMINISTIC_FLAG="--deterministic"
EXTRA_CFG_OPTIONS="" #"load_from=/home/hao1rng/core/home_fs/exp_faris/vad/orig/VAD_tiny_e2e_reproduce_a100_g2_b4_w4_exp_faris_16-06-2025_11-22-40/epoch_20.pth"

# Checkpoint Configuration
CHECKPOINT_INTERVAL=3

# ==============================================================================
# Setup Cleanup Function and Trap 
# ==============================================================================

# Create temp config 
CONFIG_PATH=$(realpath "$CONFIG_PATH")
TEMP_CONFIG_PATH="$WORK_DIR/tmp_train_config/$(basename "$CONFIG_PATH")"

# Define cleanup function
cleanup() {
    echo ""
    echo "Cleaning up..."
    if [ -f "$TEMP_CONFIG_PATH" ]; then
        rm "$TEMP_CONFIG_PATH"
        echo "Cleaned up temporary config file: $TEMP_CONFIG_PATH"
    fi
    if [ -d "$(dirname "$TEMP_CONFIG_PATH")" ]; then
        rmdir "$(dirname "$TEMP_CONFIG_PATH")" 2>/dev/null || true
    fi
}

# Set up cleanup on script exit (including interruption)
trap cleanup EXIT INT TERM QUIT HUP

# ==============================================================================
# Create Temporary Config File with Updated Settings
# ==============================================================================

mkdir -p "$(dirname "$TEMP_CONFIG_PATH")"

cat > "$TEMP_CONFIG_PATH" << EOF
# Temporary config file that extends the base config
_base_ = '$CONFIG_PATH'

# Partially override data configuration
data = dict(
    samples_per_gpu=$SAMPLES_PER_GPU,
)

# Override checkpoint configuration
checkpoint_config = dict(
    interval=$CHECKPOINT_INTERVAL,
    max_keep_ckpts=-1,
    save_optimizer=True,
)

# Override custom hooks - this will replace the entire custom_hooks list
custom_hooks = [
    dict(type='CustomSetEpochInfoHook'),
    dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=1,
        save_optimizer=True, 
        filename_tmpl='epoch_{}_inter.pth'
    )
]
EOF

# Verify the temp config file was created successfully
if [ ! -f "$TEMP_CONFIG_PATH" ]; then
    echo "Error: Failed to create temporary config file at $TEMP_CONFIG_PATH"
    exit 1
fi

echo "Temporary config file created at: $TEMP_CONFIG_PATH"

# ==============================================================================
# Build Command
# ==============================================================================

# Base command using the temporary config
CMD="python -m torch.distributed.run"
CMD="$CMD --nproc_per_node=$NGPUS"
CMD="$CMD --master_port=$MASTER_PORT"
CMD="$CMD tools/train.py"
CMD="$CMD $TEMP_CONFIG_PATH"
CMD="$CMD --launcher $LAUNCHER"
CMD="$CMD $DETERMINISTIC_FLAG"
CMD="$CMD --work-dir $WORK_DIR"

# Add any extra config options if needed
if [ ! -z "$EXTRA_CFG_OPTIONS" ]; then
    CMD="$CMD --cfg-options $EXTRA_CFG_OPTIONS"
fi

# ==============================================================================
# Execute
# ==============================================================================

echo "======================================"
echo "Training Launch Configuration"
echo "======================================"
echo "Base Config: $CONFIG_PATH"
echo "Temp Config: $TEMP_CONFIG_PATH"
echo "Work Dir: $WORK_DIR"
echo "======================================"
echo "Command:"
echo "$CMD"
echo "======================================"

echo "Starting training..."

# Execute the command
eval $CMD

echo "Training completed normally."