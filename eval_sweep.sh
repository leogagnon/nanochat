#!/bin/bash
# Evaluate all checkpoints of all runs belonging to a sweep.
# Usage: eval_sweep.sh <sweep_id> [extra eval.py args...]
#
# Example:
#   eval_sweep.sh scaling_law_sanity
#   eval_sweep.sh scaling_law_sanity max_per_task=500

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <sweep_id> [extra eval.py args...]"
    exit 1
fi

SWEEP_ID="$1"
shift

CKPTS=$(python -m scripts.find_sweep_checkpoints "$SWEEP_ID")

if [ -z "$CKPTS" ]; then
    echo "No checkpoints found for sweep_id=$SWEEP_ID"
    exit 1
fi

# Join ckpts with commas for Hydra multirun
CKPTS_CSV=$(echo "$CKPTS" | paste -sd ',')

echo "Found checkpoints: $CKPTS_CSV"
python eval.py -m "ckpt=$CKPTS_CSV" "$@"
