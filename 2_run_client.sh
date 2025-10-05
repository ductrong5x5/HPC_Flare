#!/bin/bash

# Base directory
BASE_DIR="./example_intranode"

# Check if BASE_DIR exists
if [ ! -d "$BASE_DIR" ]; then
    echo "Error: Directory $BASE_DIR does not exist."
    exit 1
fi

cd "$BASE_DIR" || exit 1

# Ask total number of clients
read -p "Enter total number of clients: " TOTAL_CLIENTS

# Activate virtual environment
if [ -f "../python_env/env/bin/activate" ]; then
    source ../python_env/env/bin/activate
else
    echo "Error: Virtual environment not found."
    exit 1
fi

# Loop through sites from 1 to TOTAL_CLIENTS
for ((i=1; i<=TOTAL_CLIENTS; i++)); do
    SITE_DIR="site-$i"
    START_SCRIPT="$SITE_DIR/startup/start.sh"

    # Compute GPU index (8 clients per GPU)
    GPU=$(( (i - 1) / 8 ))

    if [ -f "$START_SCRIPT" ]; then
        echo "Starting $SITE_DIR on GPU $GPU ..."
        CUDA_VISIBLE_DEVICES=$GPU bash "$START_SCRIPT" &
    else
        echo "Warning: $START_SCRIPT not found, skipping $SITE_DIR..."
    fi
done

# Wait for all background jobs to finish
wait
echo "All client sites finished."
