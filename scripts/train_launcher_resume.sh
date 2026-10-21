#!/bin/bash

# Script to launch training from resumed experiment.

#########################################################################
# Define functions
#########################################################################

function train_vad {
    sed "s|CONDA_ENV_NAME|${CONDA_ENV_NAME}|g; \
        s|MASTER_PORT|${MASTER_PORT}|g; \
        s|QUEUE|${QUEUE}|g; \
        s|CONFIG_PATH|${CONFIG_PATH}|g; \
        s|NUM_GPUS|${NUM_GPUS}|g; \
        s|BATCH_PER_GPU|${BATCH_PER_GPU}|g; \
        s|RUN_NR|${RUN_NR}|g; s|RUN_NAME|${RUN_NAME}|g; \
        s|SAVE_PATH|${SAVE_PATH}|g; \
        s|WALLCLOCK_TIME|${WALLCLOCK_TIME}|g; \
        s|RAM|${RAM}|g; \
        s|MIN_CPU_CORES|${MIN_CPU_CORES}|g; \
        s|WORKERS_PER_GPU|${WORKERS_PER_GPU}|g; \
        s|SAVE_LOG_DIR|${SAVE_LOG_DIR}|g; \
        s|OTHER_ARGS|${OTHER_ARGS}|g; \
        s|ROOT_DIR|${ROOT_DIR}|g; "< ${BSUB_SCRIPT} | bsub 
}

# function to find the experiment folder based on the current path
function find_basedir {
    local current_path="$1"
    
    # Check if current path contains resume_ pattern
    if [[ "$current_path" == *"/resume_"* ]]; then
        # Extract everything before the first resume_ occurrence
        echo "$current_path" | sed 's|/resume_.*||'
    else
        # No resume_ pattern found, this is the base directory
        echo "$current_path"
    fi
}

# function to log run info in JSON format
function create_run_info_json {
    local json_file="$1"
    local config_path="$2"
    local save_path="$3"
    local run_name="$4"
    local other_args="$5"
    
    cat > "$json_file" << EOF
{
    "run_info": {
        "config_path": "$config_path",
        "save_path": "$save_path",
        "run_name": "$run_name",
        "other_args": "$other_args"
    },
    "training_config": {
        "num_gpus": $NUM_GPUS,
        "batch_per_gpu": $BATCH_PER_GPU,
        "workers_per_gpu": $WORKERS_PER_GPU,
        "queue": "$QUEUE",
        "wallclock_time": "$WALLCLOCK_TIME",
        "ram": $RAM,
        "min_cpu_cores": $MIN_CPU_CORES,
        "conda_env": "$CONDA_ENV_NAME"
    },
    "resume_config": {
        "resume_folder": "$RESUME_FOLDER",
        "epoch_to_resume": "$EPOCH_TO_RESUME"
    }
}
EOF
}

#########################################################################
# Define Parameters
#########################################################################

# General settings
timestamp=$(date +"%d-%m-%Y_%H-%M-%S")
BSUB_SCRIPT=./scripts/train_vad_template.bsub
MASTER_PORT=$((RANDOM % 9761 + 10240)) 
CONDA_ENV_NAME=vad_vlp-gcc              # CHANGE

# Resources-related
QUEUE=batch_a100
NUM_GPUS=2
BATCH_PER_GPU=4
WORKERS_PER_GPU=4                      
WALLCLOCK_TIME="130:00"
RAM=64000
MIN_CPU_CORES=4

# Run-related
ROOT_DIR=~/core/VAD             # CHANGE: path to your VAD repo

RESUME_FOLDER="."   # "." if resuming from the base directory, or "resumed_..." if resuming from a specific folder
EPOCH_TO_RESUME=latest.pth 

# absolute paths to config in your exps*/ folder 
CONFIG_PATH=/home/hao1rng/core/home_fs/exp_andreas/vad-fixed/probabilistic/slp_egoagent/multivar_normal_mlp/indep_var_mu_baseline/heavy/w1e-2/VAD_tiny_e2e_slp-egoagent_multivar-normal-mlp_indep-var-mu-baseline_heavy_w1e-2_a100_g2_b4_w4_exp_andreas_02-09-2025_10-24-20/VAD_tiny_e2e_slp-egoagent_multivar-normal-mlp_indep-var-mu-baseline_heavy_w1e-2.py

CONFIG_PATH=$(realpath "$CONFIG_PATH")
PREFIX_SUBFOLDER_STRUCTURE="exp_andreas"
CUSTOM_EPOCH_SAVING=true                    # save every 5th epoch

##########################################################################
# Prepare Environment and Launch Training
##########################################################################

# Check if PREFIX_SUBFOLDER_STRUCTURE/ exists in path and throw error if not (needed for correct save path)
if [[ "$CONFIG_PATH" != *"$PREFIX_SUBFOLDER_STRUCTURE/"* ]]; then
    echo "Error: Path must contain $PREFIX_SUBFOLDER_STRUCTURE directory"
    exit 1
fi

# Extract path after PREFIX_SUBFOLDER_STRUCTURE/
CONFIG_SUBFOLDERS=$(echo "$CONFIG_PATH" | awk -F"$PREFIX_SUBFOLDER_STRUCTURE/" '{if (NF > 1) print $2}' | sed 's|/[^/]*$||')

# Find the base directory (where original training data was saved)
CONFIG_DIR=$(dirname "$CONFIG_PATH")
BASE_DIR=$(find_basedir "$CONFIG_DIR")


# Determine source directory based on RESUME_FOLDER setting
if [[ "$RESUME_FOLDER" == "." ]]; then
    SOURCE_DIR="$BASE_DIR"
    echo "Mode: Resuming from base directory (first time resume)"
else
    SOURCE_DIR="$BASE_DIR/$RESUME_FOLDER"
    if [[ ! -d "$SOURCE_DIR" ]]; then
        echo "Error: Resume folder '$RESUME_FOLDER' not found in base directory"
        echo "Available folders in $BASE_DIR:"
        ls -la "$BASE_DIR" | grep "resumed_"
        exit 1
    fi
    echo "Mode: Resuming from existing resume folder '$RESUME_FOLDER'"
fi

RUN_NAME=$(basename "$BASE_DIR")_resume_${timestamp}
SAVE_PATH="${BASE_DIR}/resumed_${timestamp}"
SAVE_LOG_DIR=${SAVE_PATH}

echo "Resuming training from: $SOURCE_DIR"
echo "Saving resumed training to: $SAVE_PATH"

CHECKPOINT_PATH="${SOURCE_DIR}/${EPOCH_TO_RESUME}"
if [[ ! -f "$CHECKPOINT_PATH" ]]; then
    echo "Error: Checkpoint not found at $CHECKPOINT_PATH"
    exit 1
fi

OTHER_ARGS="resume_from=$(realpath ${CHECKPOINT_PATH})" 

mkdir -p ${SAVE_LOG_DIR}    # otherwise bsub will not log

# Create run info JSON file
JSON_FILE="${SAVE_LOG_DIR}/run_info.json"
create_run_info_json "$JSON_FILE" "$CONFIG_PATH" "$SAVE_PATH" "$RUN_NAME" "$OTHER_ARGS"

##########################################################################
# Modify original config file via creating tmp cfg file (if needed)
##########################################################################

if [ "$CUSTOM_EPOCH_SAVING" = "true" ]; then
    # Create temp config 
    TEMP_CONFIG_PATH="${SAVE_LOG_DIR}/tmp_cfg/$(basename "$CONFIG_PATH")"

    # Create Temporary Config File with Updated Settings
    mkdir -p "$(dirname "$TEMP_CONFIG_PATH")"

    cat > "$TEMP_CONFIG_PATH" << EOF
# Temporary config file that extends the base config
_base_ = '$CONFIG_PATH'

# Override checkpoint configuration
checkpoint_config = dict(
    interval=5,
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
        echo "Error: Failed to create updated temporary config file at $TEMP_CONFIG_PATH"
        exit 1
    fi

    echo "Temporary updated config file created at: $TEMP_CONFIG_PATH"

    # Update CONFIG_PATH
    CONFIG_PATH="$TEMP_CONFIG_PATH"

fi 
##########################################################################
# Launch training
##########################################################################

train_vad
