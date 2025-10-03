#!/bin/bash
# ===== SLURM Configuration =====
#SBATCH --job-name=test_nvflare
#SBATCH -A csc666
#SBATCH --qos=normal 
#SBATCH --partition=batch
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --nodes=2
#SBATCH --ntasks=9
export CLIENTS_PER_NODE=8
# ===== Client Configuration: set once here =====
# export NUM_CLIENTS=$(( SLURM_NTASKS - 1 ))
export NUM_CLIENTS=8
export NUM_NODES=$(( (NUM_CLIENTS + CLIENTS_PER_NODE - 1) / CLIENTS_PER_NODE ))
export TOTAL_TASKS=$((NUM_CLIENTS + 1)) 
export SLURM_JOB_NAME=$SLURM_JOB_NAME


# ===== Job Parameters =====
# Set the log level for this run
export LOG_LEVEL=INFO   # or INFO or DEBUG_CLIENT or DEBUG_SERVER or DEBUG_ALL
export JOB_NAME=bert_ncbi_gaussian_8
export SYSTEM_NAME=frontier
export ASSIGN_FOLDER=Frontier
export LOCATION=/lustre/orion/csc666/proj-shared/ducnguyen/new_nvflare_edge/AMD_NVFlare
export JOB_FOLDER=$LOCATION/experiment_folder/job/${JOB_NAME}
export Total_clients=$NUM_CLIENTS
``
# GPU Configuration
export GPU_ASSIGN_MODE=2
export NUMBER_AVAIL_GPU=8
export MEM_EACH_GPU=60
export GPU_PER_CLIENT=1
export CLIENTS_PER_GPU=8

# ===== Example name and environment =====
NAME="${SLURM_JOB_ID}-$(date +'%y_%m_%d')-${USER}-${SLURM_JOB_NAME}-${JOB_NAME}-GPU_MODE_${GPU_ASSIGN_MODE}"
export EXAMPLE_NAME=$NAME 

# ===== Module Load Section =====
module load PrgEnv-gnu/8.5.0
module load miniforge3/23.11.0-0
module load rocm/5.6.0
module load craype-accel-amd-gfx90a

# ===== Use dynamic Conda environment path based on script location =====
source $LOCATION/python_env/env/bin/activate


# ===== Hugging Face cache config =====
export HF_HOME=$LOCATION/hf_home
export HF_HUB_DISABLE_TELEMETRY=1
mkdir -p "$HF_HOME"

# Bypass ROCm MIOpen disk I/O errors - Frontier
export MIOPEN_USER_DB_PATH="/tmp/my-miopen-cache"
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
rm -rf ${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}

mkdir -p logs

# ===== Decide GPU count for server =====
case "$GPU_ASSIGN_MODE" in
    1) GPU_COUNT="$GPU_PER_CLIENT" ;;
    2) GPU_COUNT=1 ;;
    3) GPU_COUNT="$NUMBER_AVAIL_GPU" ;;
esac

# ===== Launch server (same node as clients) =====
SERVER_READY_FILE="/tmp/server_ready_${SLURM_JOB_ID}.flag"
rm -f "$SERVER_READY_FILE"

srun --ntasks=1 --nodes=1  --cpus-per-task=1 setup.sh $NAME frontier server $GPU_COUNT $MEM_EACH_GPU "$SERVER_READY_FILE"  &

echo "Waiting for server to signal readiness..."
while [ ! -f "$SERVER_READY_FILE" ]; do
    sleep 10
done
echo "Server is ready. Starting client srun..."
# ===== Launch clients =====
#8 clients per node, 1 gpu per clients
srun --ntasks=$NUM_CLIENTS --nodes=1 --ntasks-per-node=$CLIENTS_PER_NODE --gpus-per-task=1  --gpu-bind=closest setup.sh $NAME frontier client &
 
# # ===== Post-processing =====
sleep 60s
echo "All NVFlare tasks completed. Running post-processing script..." 
python start_admin.py 

sleep 60s
echo "Python job finished. Cancelling SLURM job $SLURM_JOB_ID..." 
scancel $SLURM_JOB_ID
