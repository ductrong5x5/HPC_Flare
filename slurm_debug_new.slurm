#!/bin/bash
# ===== SLURM Configuration =====
#SBATCH --job-name=test_nvflare
#SBATCH -A csc666
#SBATCH --qos=normal 
#SBATCH --partition=batch
#SBATCH --time=00:15:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --nodes=2
#SBATCH --ntasks=9
export CLIENTS_PER_NODE=8
# ===== Client Configuration: set once here =====
export NUM_CLIENTS=$(( SLURM_NTASKS - 1 ))  #No change
# export NUM_NODES=$(( (NUM_CLIENTS + CLIENTS_PER_NODE - 1) / CLIENTS_PER_NODE ))   #No change
export TOTAL_TASKS=$((NUM_CLIENTS + 1))     #No change
export SLURM_JOB_NAME=$SLURM_JOB_NAME       #No change


# ===== Job Parameters =====
# Set the log level for this run
export LOG_LEVEL=INFO   # or INFO or DEBUG_CLIENT or DEBUG_SERVER or DEBUG_ALL 
export JOB_NAME=bert_ncbi_gaussian_8        #change
export SYSTEM_NAME=frontier             #Optional
export ASSIGN_FOLDER=Frontier           #Optional
export LOCATION=/lustre/orion/csc666/proj-shared/ducnguyen/new_nvflare_edge/AMD_NVFlare                             #change
export JOB_FOLDER=$LOCATION/experiment_folder/job/${JOB_NAME}   #No change
export Total_clients=$NUM_CLIENTS                   #No change
``
# GPU Configuration
export GPU_ASSIGN_MODE=2            #No change
export NUMBER_AVAIL_GPU=8           #No change 
export MEM_EACH_GPU=60
export GPU_PER_CLIENT=1             #No change
export CLIENTS_PER_GPU=8            #No change

# ===== Example name and environment =====
NAME="${SLURM_JOB_ID}-$(date +'%y_%m_%d')-${USER}-${SLURM_JOB_NAME}-${JOB_NAME}-GPU_MODE_${GPU_ASSIGN_MODE}"                #No change
export EXAMPLE_NAME=$NAME                           #No change

# ===== Module Load Section =====
module load PrgEnv-gnu/8.5.0                    #Change
module load miniforge3/23.11.0-0                #Change
module load rocm/5.6.0                          #Change
module load craype-accel-amd-gfx90a             #Change

# ===== Use dynamic Conda environment path based on script location =====
source $LOCATION/python_env/env/bin/activate                #change


# ===== Hugging Face cache config =====
export HF_HOME=$LOCATION/hf_home                    #Optional
export HF_HUB_DISABLE_TELEMETRY=1                   #Optional
mkdir -p "$HF_HOME"

# Bypass ROCm MIOpen disk I/O errors - Frontier
export MIOPEN_USER_DB_PATH="/tmp/my-miopen-cache"       #Optional
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}   #Optional
rm -rf ${MIOPEN_USER_DB_PATH}                           #Optional   
mkdir -p ${MIOPEN_USER_DB_PATH}                         #Optional

mkdir -p logs                         #No change    

# ===== Decide GPU count for server ===== No change
case "$GPU_ASSIGN_MODE" in
    1) GPU_COUNT="$GPU_PER_CLIENT" ;;
    2) GPU_COUNT=1 ;;
    3) GPU_COUNT="$NUMBER_AVAIL_GPU" ;;
esac

# ===== Launch server (same node as clients) =====
SERVER_READY_FILE="/tmp/server_ready_${SLURM_JOB_ID}.flag"  #No change
rm -f "$SERVER_READY_FILE"                     #No change

srun --ntasks=1 --nodes=1  --cpus-per-task=1 --cpu-bind=threads setup.sh $NAME frontier server $GPU_COUNT $MEM_EACH_GPU "$SERVER_READY_FILE"  &        #No change

echo "Waiting for server to signal readiness..."
while [ ! -f "$SERVER_READY_FILE" ]; do                 #No change
    sleep 10
done            
echo "Server is ready. Starting client srun..."
# ===== Launch clients =====
#8 clients per node, 8 clients per gpu
srun --ntasks=$NUM_CLIENTS --nodes=1 --ntasks-per-gpu=8  --gpu-bind=closest setup.sh $NAME frontier client &         #No change
 
# # ===== Post-processing =====
sleep 60s
echo "All NVFlare tasks completed. Running post-processing script..." 
python start_admin.py               #No change

sleep 60s
echo "Python job finished. Cancelling SLURM job $SLURM_JOB_ID..." 
scancel $SLURM_JOB_ID               #No change
