#!/bin/bash

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
        s|RUN_NAME|${RUN_NAME}|g; \
        s|SAVE_PATH|${SAVE_PATH}|g; \
        s|WALLCLOCK_TIME|${WALLCLOCK_TIME}|g; \
        s|RAM|${RAM}|g; \
        s|MIN_CPU_CORES|${MIN_CPU_CORES}|g; \
        s|WORKERS_PER_GPU|${WORKERS_PER_GPU}|g; \
        s|SAVE_LOG_DIR|${SAVE_LOG_DIR}|g; \
        s|OTHER_ARGS|${OTHER_ARGS}|g; \
        s|ROOT_DIR|${ROOT_DIR}|g; "< ${BSUB_SCRIPT} | bsub 
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
CONDA_ENV_NAME=vad_vlp           # CHANGE

# Resources-related
QUEUE=batch_a100
NUM_GPUS=2
BATCH_PER_GPU=4
WORKERS_PER_GPU=4                      
WALLCLOCK_TIME="144:00"
RAM=64000
MIN_CPU_CORES=4

# Run-related
ROOT_DIR=/home/mju2abt/workspace/Our-VAD/            # CHANGE: path to your VAD repo


# relative paths to config
# Launch this config TWICE
CONFIG_PATH=vlp/vlp/configs/vad-fixed/probabilistic/slp_egoagent/multivar_normal_mlp/indep_var_mu_baseline/from_query/heavy/s6/w1e-1_freezepostgrad/VAD_tiny_e2e_slp-egoagent_multivar-normal-mlp_indep-var-mu-baseline_from_Q_heavy_s6_w1e-1_freezepostgrad.py

CONFIG_PATH=$(realpath "$CONFIG_PATH")
PREFIX_SUBFOLDER_STRUCTURE="configs"      # CHANGE
CUSTOM_EPOCH_SAVING=true   
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
CONFIG_NAME=$(basename "$CONFIG_PATH" .py)   

RUN_NAME=${CONFIG_NAME}_${QUEUE#*_}_g${NUM_GPUS}_b${BATCH_PER_GPU}_w${WORKERS_PER_GPU}_exp_juergen_${timestamp}

SAVE_PATH=/fs/scratch/rb_bd_dlp_rng-dl01_cr_AID_employees/students/hao1rng/exp_juergen/${CONFIG_SUBFOLDERS}/${RUN_NAME}
SAVE_LOG_DIR=${SAVE_PATH}

echo "======================================"
echo "Run config: $CONFIG_PATH"
echo "Save training run to: $SAVE_PATH"

OTHER_ARGS="" 

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
 