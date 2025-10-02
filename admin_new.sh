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
READY_FILE=${4:-}
MAX_PROCS=${MAX_PROCS:-4}   # default parallel processes

if [ -z "$EXAMPLE_SUBDIR" ] || [ -z "$NUM_GPUS" ] || [ -z "$MEM_PER_GPU" ]; then
    echo "Usage: $0 <example_dir> <num_gpus> <mem_per_gpu_in_GiB> [ready_file]"
    exit 1
fi

echo " Using example directory: $EXAMPLE_SUBDIR"
mkdir -p "$EXAMPLE_SUBDIR"

PARENT_DIR=$(dirname "$EXAMPLE_SUBDIR")
cd "$PARENT_DIR" || exit 1

# Show IPv4
echo "=====> Finding IP Address"
ip -4 a | grep "inet"
echo

# Load environment modules
module load PrgEnv-gnu/8.5.0
module load miniforge3/23.11.0-0
module load rocm/5.6.0
module load craype-accel-amd-gfx90a

# echo "Using Conda env: $CONDA_ENV_NAME"
# export PATH="${CONDA_ENV_PATH}/bin:$PATH"
# source activate "${CONDA_ENV_PATH}"
source $LOCATION/python_env/env/bin/activate

# Create & modify project.yml
echo "=====> Creating & Modifying project.yml"
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

# Provision again
nvflare provision -p project.yml

# chmod *.sh in parallel
echo "=====> Setting Execute Permissions"
find . -type f -name "*.sh" -print0 | xargs -0 -P"$MAX_PROCS" chmod +x

# Remove '&' from startup scripts
echo "=====> Fixing startup.sh"
sed -i 's|^[[:space:]]*\$DIR/sub_start\.sh[[:space:]]*&[[:space:]]*$|  $DIR/sub_start.sh|' \
  "workspace/test/prod_00/$server_name/startup/start.sh"

for i in $(seq 1 ${Total_clients:-0}); do
    CLIENT_NAME="site-$i"
    sed -i 's|^[[:space:]]*\$DIR/sub_start\.sh[[:space:]]*&[[:space:]]*$|  $DIR/sub_start.sh|' \
      "workspace/test/prod_00/$CLIENT_NAME/startup/start.sh"
done

# Copy workspace (parallelized)
echo "=====> Copying project folders"
TARGET_DIR=$(basename "$EXAMPLE_SUBDIR")
printf "%s\n" \
  "workspace/test/prod_00/admin@ornl.gov" \
  "workspace/test/prod_00/$server_name" \
  $(seq 1 ${Total_clients:-0} | sed "s|^|workspace/test/prod_00/site-|") \
  | xargs -n1 -P"$MAX_PROCS" -I{} cp -r {} "$TARGET_DIR"
echo " Project folder copied."

# Copy Job Folder
echo "=====> Copy Job Folder"
cp -r "$JOB_FOLDER" "frontier/admin@ornl.gov/transfer/"

# Copy PT to Server and Clients (parallelized)
echo "=====> Copy custom code"
ls -d $SYSTEM_NAME/* | xargs -n1 -P"$MAX_PROCS" -I{} cp -r "$JOB_FOLDER/../../code/custom" {}

# GPU resources setup (parallelized)
echo "=====> Updating GPU resources"
ls -d $SYSTEM_NAME/site-*/local | xargs -n1 -P"$MAX_PROCS" -I{} bash -c '
  dir="{}"
  if [ -d "$dir" ]; then
    jq ".components |= map(if .id == \"resocurce_manager\" 
         then .args.num_of_gpus='$NUM_GPUS' 
         | .args.mem_per_gpu_in_GiB='$MEM_PER_GPU' else . end)" \
         "$dir/resources.json.default" > "$dir/resources.json"
    echo "Updated GPU config in $dir"
  fi
'

# comm_config.json
cat > comm_config.json <<'EOF'
{ "use_aio_grpc": true }
EOF
ls -d $SYSTEM_NAME/*/local | xargs -n1 -P"$MAX_PROCS" -I{} cp comm_config.json {}

# Max clients
SERVER_DIR="$SYSTEM_NAME/$server_name/local"
if [ -f "$SERVER_DIR/resources.json.default" ]; then
    sed 's/"max_num_clients": 100/"max_num_clients": 1000/' \
        "$SERVER_DIR/resources.json.default" > "$SERVER_DIR/resources.json"
fi

#===== Setting log level =====
case "${LOG_LEVEL:-INFO}" in
  DEBUG_CLIENT)
    echo "🔧 Applying DEBUG log level to clients only..."
    for dir in "$SYSTEM_NAME"/site-*/local; do
        if [ -d "$dir" ]; then
            cp "$dir/log_config.json.default" "$dir/log_config.json"
            jq --arg LEVEL "DEBUG" '
              .handlers |= with_entries(.value.level = $LEVEL)
              | .loggers.root.level = $LEVEL
            ' "$dir/log_config.json" > "$dir/log_config_tmp.json" \
              && mv "$dir/log_config_tmp.json" "$dir/log_config.json"
            echo "✅ Updated log_config.json in $dir"
        fi
    done
    ;;
  DEBUG_SERVER)
    echo "🔧 Applying DEBUG log level to server only..."
    cp "$SYSTEM_NAME/$server_name/local/log_config.json.default" "$SYSTEM_NAME/$server_name/local/log_config.json"
    jq --arg LEVEL "DEBUG" '
      .handlers |= with_entries(.value.level = $LEVEL)
      | .loggers.root.level = $LEVEL
    ' "$SYSTEM_NAME/$server_name/local/log_config.json" > "$SYSTEM_NAME/$server_name/local/log_config_tmp.json" \
      && mv "$SYSTEM_NAME/$server_name/local/log_config_tmp.json" "$SYSTEM_NAME/$server_name/local/log_config.json"
    echo "✅ Updated log_config.json for server: $server_name"
    ;;
  DEBUG_ALL)
    echo "🔧 Applying DEBUG log level to both clients and server..."
    for dir in "$SYSTEM_NAME"/site-*/local; do
        if [ -d "$dir" ]; then
            cp "$dir/log_config.json.default" "$dir/log_config.json"
            jq --arg LEVEL "DEBUG" '
              .handlers |= with_entries(.value.level = $LEVEL)
              | .loggers.root.level = $LEVEL
            ' "$dir/log_config.json" > "$dir/log_config_tmp.json" \
              && mv "$dir/log_config_tmp.json" "$dir/log_config.json"
            echo "✅ Updated log_config.json in $dir"
        fi
    done
    cp "$SYSTEM_NAME/$server_name/local/log_config.json.default" "$SYSTEM_NAME/$server_name/local/log_config.json"
    jq --arg LEVEL "DEBUG" '
      .handlers |= with_entries(.value.level = $LEVEL)
      | .loggers.root.level = $LEVEL
    ' "$SYSTEM_NAME/$server_name/local/log_config.json" > "$SYSTEM_NAME/$server_name/local/log_config_tmp.json" \
      && mv "$SYSTEM_NAME/$server_name/local/log_config_tmp.json" "$SYSTEM_NAME/$server_name/local/log_config.json"
    echo "✅ Updated log_config.json for server: $server_name"
    ;;
  INFO|"")
    echo "ℹ️ LOG_LEVEL=$LOG_LEVEL → leaving log_config.json unchanged"
    ;;
  *)
    echo "⚠️ Unknown LOG_LEVEL=$LOG_LEVEL, expected INFO | DEBUG_CLIENT | DEBUG_SERVER | DEBUG_ALL"
    ;;
esac

echo "==================================================="
echo "✅ Multiprocess Setup Completed!"
echo "==================================================="

if [ -n "$READY_FILE" ]; then
    touch "$READY_FILE"
fi
