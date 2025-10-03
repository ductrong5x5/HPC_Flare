#!/bin/bash

module load PrgEnv-gnu/8.5.0
module load miniforge3/23.11.0-0
module load rocm/5.6.0
module load craype-accel-amd-gfx90a
ROLE=$1
# Get SLURM rank
RANK=${SLURM_PROCID:-0}
NUM_TASKS=${SLURM_NTASKS:-1}

# Use env vars
example_name="${EXAMPLE_NAME}"
system_name="${SYSTEM_NAME}"
Client_per_GPU="${CLIENTS_PER_GPU}"
BASE_DIR="${LOCATION}/example/${SLURM_JOB_NAME}/${example_name}/${system_name}"
PARENT_DIR=$(dirname "$BASE_DIR")


# Check BASE_DIR
if [ ! -d "$BASE_DIR" ]; then
    echo "Error: Directory $BASE_DIR does not exist."
    echo
    exit 1
fi

# Change to parent directory
cd "$PARENT_DIR" || exit 1
cd "$BASE_DIR" || exit 1

# Create log folder for this task
LOG_DIR="$BASE_DIR/logs/task_$RANK"
mkdir -p "$LOG_DIR"


# echo "Using Conda env: $CONDA_ENV_NAME"
# echo "Activating from: $CONDA_ENV_PATH"

# export PATH="${CONDA_ENV_PATH}/bin:$PATH"
# source activate "${CONDA_ENV_PATH}"
source $LOCATION/python_env/env/bin/activate
if [ "$ROLE" = "server" ]; then
    # SET Proxy to download model from compute nodes
    export all_proxy=socks://proxy.ccs.ornl.gov:3128/
    export ftp_proxy=ftp://proxy.ccs.ornl.gov:3128/
    export http_proxy=http://proxy.ccs.ornl.gov:3128/
    export https_proxy=http://proxy.ccs.ornl.gov:3128/
    export no_proxy='localhost,127.0.0.0/8,*.ccs.ornl.gov'

    # Start main nvflare server
    echo " Task $RANK will handle the main server"

    # Get the node name
    server_name=$(hostname)
    # server_name="localhost" #intranode

    # Setup the paths for the start script
    START_SCRIPT="$server_name/startup/start.sh"
    LOG_FILE="$LOG_DIR/$server_name.log"

    # Launch server
    if [ -f "$START_SCRIPT" ]; then
        echo " Launching $SITE_DIR (logging to $LOG_FILE)"
        bash "$START_SCRIPT" > "$LOG_FILE" 2>&1 &
    else
        echo " $START_SCRIPT not found. Skipping site-$i." | tee "$LOG_FILE"
    fi
else
    # Start the clients
    RANK_PLUS_ONE=$((RANK + 1))
    echo " Task $RANK will handle the site-$RANK_PLUS_ONE"
    echo

    SITE_DIR="site-$RANK_PLUS_ONE"
    START_SCRIPT="$SITE_DIR/startup/start.sh"
    LOG_FILE="$LOG_DIR/site_$RANK_PLUS_ONE.log"

    echo " Checking $START_SCRIPT..."
    if [ -f "$START_SCRIPT" ]; then
        echo " Launching $SITE_DIR (logging to $LOG_FILE)"
        bash "$START_SCRIPT" > "$LOG_FILE" 2>&1 &
    else
        echo " $START_SCRIPT not found. Skipping site-$i." | tee "$LOG_FILE"
    fi
    
    unset all_proxy
    unset ftp_proxy
    unset http_proxy
    unset https_proxy
    unset no_proxy
fi

wait  # Ensure background jobs complete before script exits
