#!/bin/bash
set -euo pipefail

# Base transfer directory
BASE_DIR="./example_intranode/admin@ornl.gov/transfer"

# Check LOCATION is set
if [ -z "${LOCATION:-}" ]; then
    echo "Error: LOCATION environment variable is not set."
    exit 1
fi

# Loop over all job folders
for JOB_DIR in "$BASE_DIR"/*/; do
    JOB_NAME=$(basename "$JOB_DIR")
    CONFIG_FILE="$JOB_DIR/app/config/config_fed_client.json"

    if [ -f "$CONFIG_FILE" ]; then
        echo "Updating DATASET_ROOT in $CONFIG_FILE ..."
        jq --arg loc "$LOCATION" '.DATASET_ROOT = "\($loc)/experiment_folder/data/nlp-ner"' \
            "$CONFIG_FILE" > "$CONFIG_FILE.tmp" && mv "$CONFIG_FILE.tmp" "$CONFIG_FILE"
    else
        echo "Warning: $CONFIG_FILE not found, skipping $JOB_NAME"
    fi
done

echo "All config_fed_client.json files updated."
