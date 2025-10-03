#!/bin/bash
set -euo pipefail

ROLE=${3:-client}
RANK=${SLURM_PROCID:-0}

echo "Running $ROLE on $(hostname) [rank: $RANK]"

case "$ROLE" in
  server)
    echo "Rank $RANK is setting up server..."

    # Help option
    if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
        cat <<EOF
Usage: $0 <example_name> <system_name> [role] [extra args...]

Example: $0 cifar10 frontier server
EOF
        exit 0
    fi

    # Argument check
    if [ $# -lt 2 ]; then
        echo "Missing arguments. Run: $0 --help"
        exit 1
    fi

    # Setup directories
    JOB_DIR="example/$SLURM_JOB_NAME"
    SYSTEM_SUBDIR="$JOB_DIR/$1/$2"
    mkdir -p "$SYSTEM_SUBDIR"

    # Run admin setup in background
    bash admin_new.sh "$SYSTEM_SUBDIR" "${4:-}" "${5:-}" "${6:-}" &

    # Start server after setup
    wait
    bash start.sh server
    ;;
  
  client)
    bash start.sh client "${4:-}"
    ;;

  *)
    echo "Unknown role: $ROLE (expected server|client)"
    exit 1
    ;;
esac
