#!/bin/bash
set -euo pipefail

echo
echo "======================================================================"
echo " Admin Script Setup (Multiprocess Optimized)"
echo "======================================================================"
echo

EXAMPLE_SUBDIR=$1
NUM_GPUS=$2
MEM_PER_GPU=$3
echo "NUM_GPUS=$NUM_GPUS"
echo "MEM_PER_GPU=$MEM_PER_GPU"
READY_FILE=${4:-}
MAX_PROCS=${MAX_PROCS:-4}   # default parallel processes

if [ -z "$EXAMPLE_SUBDIR" ] || [ -z "$NUM_GPUS" ] || [ -z "$MEM_PER_GPU" ]; then
    echo "Usage: $0 <example_dir> <num_gpus> <mem_per_gpu_in_GiB> [ready_file]"
    exit 1
fi

mkdir -p "$EXAMPLE_SUBDIR"
PARENT_DIR=$(dirname "$EXAMPLE_SUBDIR")
cd "$PARENT_DIR" || exit 1

# Show IPv4
ip -4 a | grep "inet"

# Load environment modules
module load PrgEnv-gnu/8.5.0
module load miniforge3/23.11.0-0
module load rocm/5.6.0
module load craype-accel-amd-gfx90a

# Activate Conda
source "$LOCATION/python_env/env/bin/activate"

# Create & modify project.yml
echo "Creating & modifying project.yml"
echo "2" | nvflare provision
PROJECT_YML="project.yml"
server_name=$(hostname)

sed -i \
  -e "s/name: example_project/name: test/" \
  -e "s/name: admin@nvidia.com/name: admin@ornl.gov/" \
  -e "s/org: nvidia/org: ornl/" \
  -e "s/server1/$server_name/" \
  "$PROJECT_YML"

CLIENT_COUNT=${CLIENT_COUNT:-0}
if [ "${Total_clients:-0}" -gt 2 ]; then
    for i in $(seq 3 "$Total_clients"); do
        CLIENT_NAME="site-$i"
        sed -i "/participants:/a \ \ - name: $CLIENT_NAME\n\ \ \ \ type: client\n\ \ \ \ org: ornl" "$PROJECT_YML"
    done
fi

p_c=${p_c:-8002}
p_a=${p_a:-8003}
sed -i -e "s/8002/$p_c/" -e "s/8003/$p_a/" "$PROJECT_YML"

nvflare provision -p project.yml

# Set execute permissions in parallel
find . -type f -name "*.sh" -print0 | xargs -0 -P"$MAX_PROCS" chmod +x

# Remove '&' from startup scripts
sed -i 's|^[[:space:]]*\$DIR/sub_start\.sh[[:space:]]*&[[:space:]]*$|  $DIR/sub_start.sh|' \
    "workspace/test/prod_00/$server_name/startup/start.sh"

for i in $(seq 1 ${Total_clients:-0}); do
    CLIENT_NAME="site-$i"
    sed -i 's|^[[:space:]]*\$DIR/sub_start\.sh[[:space:]]*&[[:space:]]*$|  $DIR/sub_start.sh|' \
      "workspace/test/prod_00/$CLIENT_NAME/startup/start.sh"
done

# Copy workspace folders in parallel
TARGET_DIR=$(basename "$EXAMPLE_SUBDIR")
printf "%s\n" \
  "workspace/test/prod_00/admin@ornl.gov" \
  "workspace/test/prod_00/$server_name" \
  $(seq 1 ${Total_clients:-0} | sed "s|^|workspace/test/prod_00/site-|") \
  | xargs -n1 -P"$MAX_PROCS" -I{} cp -r {} "$TARGET_DIR"

# Copy Job Folder
cp -r "$JOB_FOLDER" "$SYSTEM_NAME/admin@ornl.gov/transfer/"
CONFIG_FILE="$SYSTEM_NAME/admin@ornl.gov/transfer/$JOB_NAME/app/config/config_fed_client.json"

jq --arg loc "$LOCATION" '.DATASET_ROOT = "\($loc)/experiment_folder/data/nlp-ner"' "$CONFIG_FILE" \
   > "$CONFIG_FILE.tmp" && mv "$CONFIG_FILE.tmp" "$CONFIG_FILE"

# Copy custom code
ls -d "$SYSTEM_NAME"/* | xargs -n1 -P"$MAX_PROCS" -I{} cp -r "$JOB_FOLDER/../../code/custom" {}

# Update GPU resources in parallel
ls -d "$SYSTEM_NAME"/site-*/local | xargs -n1 -P"$MAX_PROCS" -I{} bash -c '
  dir="{}"
  if [ -d "$dir" ]; then
    jq ".components |= map(if .id == \"resource_manager\" 
         then .args.num_of_gpus='$NUM_GPUS' 
         | .args.mem_per_gpu_in_GiB='$MEM_PER_GPU' else . end)" \
         "$dir/resources.json.default" > "$dir/resources.json"
    echo "Updated GPU config in $dir"
  fi
'

# Copy comm_config.json to all local directories
# echo '{ "use_aio_grpc": true }' > comm_config.json
# ls -d "$SYSTEM_NAME"/*/local | xargs -n1 -P"$MAX_PROCS" -I{} cp comm_config.json {}

# Update max clients on server
SERVER_DIR="$SYSTEM_NAME/$server_name/local"
if [ -f "$SERVER_DIR/resources.json.default" ]; then
    sed 's/"max_num_clients": 100/"max_num_clients": 1000/' \
        "$SERVER_DIR/resources.json.default" > "$SERVER_DIR/resources.json"
fi

# Set log level
case "${LOG_LEVEL:-INFO}" in
  DEBUG_CLIENT)
    for dir in "$SYSTEM_NAME"/site-*/local; do
        [ -d "$dir" ] || continue
        cp "$dir/log_config.json.default" "$dir/log_config.json"
        jq --arg LEVEL "DEBUG" '.handlers |= with_entries(.value.level=$LEVEL) | .loggers.root.level=$LEVEL' \
           "$dir/log_config.json" > "$dir/log_config_tmp.json" && mv "$dir/log_config_tmp.json" "$dir/log_config.json"
    done
    ;;
  DEBUG_SERVER)
    cp "$SERVER_DIR/log_config.json.default" "$SERVER_DIR/log_config.json"
    jq --arg LEVEL "DEBUG" '.handlers |= with_entries(.value.level=$LEVEL) | .loggers.root.level=$LEVEL' \
       "$SERVER_DIR/log_config.json" > "$SERVER_DIR/log_config_tmp.json" && mv "$SERVER_DIR/log_config_tmp.json" "$SERVER_DIR/log_config.json"
    ;;
  DEBUG_ALL)
    for dir in "$SYSTEM_NAME"/site-*/local; do
        [ -d "$dir" ] || continue
        cp "$dir/log_config.json.default" "$dir/log_config.json"
        jq --arg LEVEL "DEBUG" '.handlers |= with_entries(.value.level=$LEVEL) | .loggers.root.level=$LEVEL' \
           "$dir/log_config.json" > "$dir/log_config_tmp.json" && mv "$dir/log_config_tmp.json" "$dir/log_config.json"
    done
    cp "$SERVER_DIR/log_config.json.default" "$SERVER_DIR/log_config.json"
    jq --arg LEVEL "DEBUG" '.handlers |= with_entries(.value.level=$LEVEL) | .loggers.root.level=$LEVEL' \
       "$SERVER_DIR/log_config.json" > "$SERVER_DIR/log_config_tmp.json" && mv "$SERVER_DIR/log_config_tmp.json" "$SERVER_DIR/log_config.json"
    ;;
  INFO|"")
    echo "LOG_LEVEL=$LOG_LEVEL → leaving log_config.json unchanged"
    ;;
  *)
    echo "Unknown LOG_LEVEL=$LOG_LEVEL (expected INFO | DEBUG_CLIENT | DEBUG_SERVER | DEBUG_ALL)"
    ;;
esac

echo "Multiprocess Setup Completed."

# Signal ready file if provided
[ -n "$READY_FILE" ] && touch "$READY_FILE"
