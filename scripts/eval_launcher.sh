#!/bin/bash

function eval_vad {
    sed "s|CONDA_ENV_NAME|${CONDA_ENV_NAME}|g; \
        s|QUEUE|${QUEUE}|g; \
        s|WALLCLOCK_TIME|${WALLCLOCK_TIME}|g; \
        s|RAM|${RAM}|g; \
        s|MIN_CPU_CORES|${MIN_CPU_CORES}|g; \
        s|RUN_NAME|${RUN_NAME}|g; \
        s|CONFIG_PATH|${CONFIG_PATH}|g; \
        s|CKPT_PATH|${CKPT_PATH}|g; \
        s|SAVE_LOG_DIR|${SAVE_LOG_DIR}|g; \
        s|WANDB_INIT_CFG|${WANDB_INIT_CFG}|g; \
        s|EPOCH_TO_EVAL|${EPOCH_TO_EVAL}|g; \
        s|WANDB_FOLDER|${WANDB_FOLDER}|g; \
        s|LOG_WANDB|${LOG_WANDB}|g; \
        s|PROB_PIPELINE|${PROB_PIPELINE}|g; \
        s|SAMPLE_N_EGO_TRAJS|${SAMPLE_N_EGO_TRAJS}|g; \
        s|SAVE_SLP_FEATS|${SAVE_SLP_FEATS}|g; \
        s|POSTERIOR|${POSTERIOR}|g; \
        s|ROOT_DIR|${ROOT_DIR}|g; "< ${BSUB_SCRIPT} | bsub 
    }

BSUB_SCRIPT=./scripts/eval_vad_template.bsub
CONDA_ENV_NAME=vad_vlp-gcc      # CHANGE: vad_vlp-test
LOG_WANDB=false
WANDB_FOLDER=/home/hao1rng/core/VAD/wandb

# Resources-related
QUEUE=batch_a100
WALLCLOCK_TIME="01:00"
RAM=32000
MIN_CPU_CORES=4

# Run-related
ROOT_DIR=~/core/VAD                       # CHANGE: root directory of your VAD repo   
    
CONFIG_PATH=/home/hao1rng/core/home_fs/exps/vad-fixed/probabilistic/slp_egoagent/multivar_normal_mlp/indep_var_mu_baseline/heavy/w1e-4/VAD_tiny_e2e_slp-egoagent_multivar-normal-mlp_indep-var-mu-baseline_heavy_w1e-4_a100_g2_b4_w4_exp_faris_06-09-2025_11-49-38/VAD_tiny_e2e_slp-egoagent_multivar-normal-mlp_indep-var-mu-baseline_heavy_w1e-4.py
EPOCH=10,15,20,25,30,35                   # CHANGE: epochs to evaluate (can be comma separated): 5,7,10

PROB_PIPELINE=true         # whether the model is probabilistic (i.e., samples multiple future trajectories)
SAMPLE_N_EGO_TRAJS=1       # number of ego trajectories to sample during inference (for propabilistic models, just to save them for vizalization, not participate in metrics aggregation)
SAVE_SLP_FEATS=false       # whether to save SLP features and queries during testing (only for models with SLP module)
POSTERIOR=false             # whether sample from posterior during evaluation, PROB_PIPELINE must be true

INTERMEDIATE_CHECKPOINTS=false  

PREFIX_SUBFOLDER_STRUCTURE="exps"       # CHANGE
# Check if configs/ exists in path and throw error if not (needed for correct save path)
if [[ "$CONFIG_PATH" != *"$PREFIX_SUBFOLDER_STRUCTURE/"* ]]; then
    echo "Error: Path must contain '$PREFIX_SUBFOLDER_STRUCTURE/' directory"
    exit 1
fi


# Define main folder of experiment (original run + additional resumed runs)
if [[ "$(dirname $CONFIG_PATH)" == *"resumed_"* ]]; then
    EXP_METADIR=$(dirname $(dirname "$CONFIG_PATH")) 
    RESUMED_RUN=true
    EXP_SUFFIX=$(basename $(dirname "$CONFIG_PATH"))  # e.g.: resumed_07-07-2025_15-52-14
    echo "Detected resumed run: $EXP_SUFFIX"
else    
    EXP_METADIR=$(dirname "$CONFIG_PATH")
    RESUMED_RUN=false
fi  

# Create the file that tracks evaluated epochs
EPOCHS_FILE="${EXP_METADIR}/evaluated_epochs.txt"
touch "$EPOCHS_FILE"
if [ "$RESUMED_RUN" = true ]; then
    echo "${EPOCH} "$(basename "$(dirname "$CONFIG_PATH")")"" >> "$EPOCHS_FILE"
fi 

if [ "$POSTERIOR" = true ]; then
    echo "${EPOCH}_posterior" >> "$EPOCHS_FILE"
else
    echo "${EPOCH}" >> "$EPOCHS_FILE"
fi

# Extract path after PREFIX_SUBFOLDER_STRUCTURE/
CONFIG_SUBFOLDERS=$(echo "$EXP_METADIR" | awk -F"$PREFIX_SUBFOLDER_STRUCTURE/" '{if (NF > 1) print $2}' | sed 's|/[^/]*$||')  # e.g.: vad/alp_slp/llm-enc/contra

# Create save path based on config subfolders (for easier organization)
SAVE_LOG_BASEDIR=./exps/eval/${CONFIG_SUBFOLDERS}/$(basename ${EXP_METADIR}) # /eval/multimodal

if [ "$POSTERIOR" = true ]; then
    # Split CONFIG_SUBFOLDERS and insert "posterior" between first and second folder
    FIRST_FOLDER=$(echo "$CONFIG_SUBFOLDERS" | cut -d'/' -f1)
    REMAINING_FOLDERS=$(echo "$CONFIG_SUBFOLDERS" | cut -d'/' -f2-)
    CONFIG_SUBFOLDERS_MOD="${FIRST_FOLDER}/posterior/${REMAINING_FOLDERS}"
    SAVE_LOG_BASEDIR=./exps/eval/${CONFIG_SUBFOLDERS_MOD}/$(basename ${EXP_METADIR})
fi

if [ "$RESUMED_RUN" = true ]; then
    RUN_NAME_BASE="$(basename "$EXP_METADIR")_${EXP_SUFFIX}"  # e.g.: ..._resumed_07-07-2025_15-52-14
else
    RUN_NAME_BASE=$(basename ${EXP_METADIR}) 
fi

### WandB Logging Args ###
WANDB_INIT_CFG="None"  

if [ "${LOG_WANDB}" = true ]; then
    WANDB_PROJECT=VLP-eval
    WANDB_RUN_NAME=${RUN_NAME_BASE}                            # run name 
    WANDB_GROUP=${CONFIG_SUBFOLDERS}                           # group name, e.g.: vad/alp_slp/llm/contra
    WANDB_TAGS=$(echo "${WANDB_GROUP}" | tr '/' ';')
    echo "W&B logging enabled: project=${WANDB_PROJECT}, group=${WANDB_GROUP}, run_name=${WANDB_RUN_NAME}, tags=${WANDB_TAGS}"
fi

echo "Evaluate Posterior: ${POSTERIOR}"

for epoch in $(echo ${EPOCH} | tr ',' ' '); do
    echo "Evaluating epoch: ${epoch}"
    RUN_NAME=${RUN_NAME_BASE}_e${epoch}                                         # parent directory name with epoch info
    SAVE_LOG_DIR=${SAVE_LOG_BASEDIR}/${RUN_NAME}_$(date +"%d-%m-%Y_%H-%M-%S")   # save log dir with timestamp
    mkdir -p ${SAVE_LOG_DIR}                                                    # create save log dir, otherwise bsub will not log
    
    if [ "$INTERMEDIATE_CHECKPOINTS" = true ]; then
        CKPT_PATH=$(dirname ${CONFIG_PATH})/epoch_${epoch}_inter.pth
    else
        CKPT_PATH=$(dirname ${CONFIG_PATH})/epoch_${epoch}.pth   
    fi

    EPOCH_TO_EVAL=${epoch}

    if [ "${LOG_WANDB}" = true ]; then
        WANDB_INIT_CFG="project=${WANDB_PROJECT},group=${WANDB_GROUP},name=${WANDB_RUN_NAME},id=${WANDB_RUN_NAME},resume=allow,tags=\"${WANDB_TAGS}\""
    fi

    eval_vad
done
