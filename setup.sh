#!/bin/bash
set -euo pipefail

ROLE=${3:-client}
RANK=${SLURM_PROCID:-0}

echo "Running $ROLE on $(hostname) [rank: $RANK]"

case "$ROLE" in
  server)
    echo "RANK: $RANK is setting up...."

    # Help function
    if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
        cat <<EOF

=================================================
 Help Documentation
=================================================

Usage: $0 <example_name> <system_name>
Example: $0 cifar10 frontier

EOF
        exit 0
    fi

    # Argument check
    if [ $# -lt 2 ]; then
        echo "Missing arguments. Run: $0 --help"
        exit 1
    fi

    ####################################
    # Setup directories (fast)
    ####################################
    EXAMPLE_DIR="example"
    JOB_DIR="$EXAMPLE_DIR/$SLURM_JOB_NAME"
    EXAMPLE_SUBDIR="$JOB_DIR/$1"
    SYSTEM_SUBDIR="$EXAMPLE_SUBDIR/$2"

    echo "📂 Creating folder structure under: $SYSTEM_SUBDIR"
    mkdir -p "$SYSTEM_SUBDIR"

    ####################################
    # Run admin setup in parallel
    ####################################
    echo "🔧 Running admin setup..."
    # bash "admin_new.sh" "$SYSTEM_SUBDIR" "${4:-}" "${5:-}" "${6:-}" &
    bash admin_new.sh "$SYSTEM_SUBDIR" "${4:-}" "${5:-}" "${6:-}" &

    ####################################
    # Start server
    ####################################
    wait
    bash "start.sh" server
    ;;
  
  client)
    bash "start.sh" client "${4:-}"
    ;;

  *)
    echo " Unknown role: $ROLE (expected server|client)"
    exit 1
    ;;
esac
