#!/bin/bash

# Script to visualize driving trajectories

function viz_vad {
    sed "s|CONDA_ENV_NAME|${CONDA_ENV_NAME}|g; \
        s|QUEUE|${QUEUE}|g; \
        s|WALLCLOCK_TIME|${WALLCLOCK_TIME}|g; \
        s|RAM|${RAM}|g; \
        s|MIN_CPU_CORES|${MIN_CPU_CORES}|g; \
        s|RUN_NAME|${RUN_NAME}|g; \
        s|RESULTS_PATH|${RESULTS_PATH}|g; \
        s|SAVE_VIZ_DIR|${SAVE_VIZ_DIR}|g; \
        s|FRACTION|${FRACTION}|g; \
        s|ROOT_DIR|${ROOT_DIR}|g; "< ${BSUB_SCRIPT} | bsub 
    }

BSUB_SCRIPT=./scripts/viz_vad_template.bsub
CONDA_ENV_NAME=vad_vlp-gcc      # CHANGE: vad_vlp-test

# Resources-related
QUEUE=batch_a100
WALLCLOCK_TIME="10:00"
RAM=32000
MIN_CPU_CORES=4

# Run-related
ROOT_DIR=~/core/VAD                       # CHANGE: root directory of your VAD repo   

RESULTS_PATH=/home/hao1rng/core/VAD/exps/eval/multimodal/vad-fixed/probabilistic/slp_egoagent/multivar_normal_mlp/indep_var_mu_baseline/w1e-2/VAD_tiny_e2e_slp-egoagent_multivar-normal-mlp_indep-var-mu-baseline_w1e-2_a100_g2_b4_w4_exp_faris_06-09-2025_11-48-07/VAD_tiny_e2e_slp-egoagent_multivar-normal-mlp_indep-var-mu-baseline_w1e-2_a100_g2_b4_w4_exp_faris_06-09-2025_11-48-07_e30_11-09-2025_19-32-35/pts_bbox/results_nusc.pkl

FRACTION=0.3   # fraction of data to visualize [0.0, 1.0]

PREFIX_SUBFOLDER_STRUCTURE="VAD"      # CHANGE
# Check if configs/ exists in path and throw error if not (needed for correct save path)
if [[ "$RESULTS_PATH" != *"$PREFIX_SUBFOLDER_STRUCTURE/"* ]]; then
    echo "Error: Path must contain '$PREFIX_SUBFOLDER_STRUCTURE/' directory"
    exit 1
fi

# Define main folder of experiment
RUN_NAME_BASE=$(basename $(dirname $(dirname "$RESULTS_PATH")))
EXP_METADIR=$(dirname $(dirname $(dirname "$RESULTS_PATH")))

# Create the file that tracks visualized epochs
VIZ_EPOCHS_FILE="${EXP_METADIR}/viz_epochs.txt"
touch "$VIZ_EPOCHS_FILE"
echo "${RUN_NAME_BASE}" >> "$VIZ_EPOCHS_FILE"

timestamp=$(date +"%d-%m-%Y_%H-%M-%S")
SAVE_VIZ_DIR=$(dirname $(dirname "$RESULTS_PATH"))/viz_${timestamp}

RUN_NAME=${RUN_NAME_BASE}_viz
                          
mkdir -p "$SAVE_VIZ_DIR"

echo "Evaluating $(basename "$RUN_NAME_BASE")"
echo "Fraction: ${FRACTION}"
echo "Save to: ${SAVE_VIZ_DIR}"

viz_vad

