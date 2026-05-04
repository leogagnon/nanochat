#!/bin/bash
# Evaluate all checkpoints of all runs belonging to a sweep.
# Usage:
#   eval_sweep.sh <sweep_id> [--run-filter key=value ...] [--ckpt-root path] [--] [extra eval.py args...]
#
# Example:
#   eval_sweep.sh scaling_law_sanity
#   eval_sweep.sh scaling_law_sanity --run-filter model.n_layer=16
#   eval_sweep.sh scaling_law_sanity --run-filter data.train_bin=tiny_shakespeare_train.bin max_per_task=500
#   eval_sweep.sh scaling_law_sanity max_per_task=500

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <sweep_id> [--run-filter key=value ...] [--ckpt-root path] [--] [extra eval.py args...]"
    exit 1
fi

SWEEP_ID="$1"
shift

RUN_FILTERS=()
CKPT_ROOT=""
while [ $# -gt 0 ]; do
    case "$1" in
        --run-filter)
            if [ -z "${2:-}" ]; then
                echo "Missing value for --run-filter (expected key=value)"
                exit 1
            fi
            RUN_FILTERS+=("$2")
            shift 2
            ;;
        --run-filter=*)
            RUN_FILTERS+=("${1#*=}")
            shift
            ;;
        --ckpt-root)
            if [ -z "${2:-}" ]; then
                echo "Missing value for --ckpt-root"
                exit 1
            fi
            CKPT_ROOT="$2"
            shift 2
            ;;
        --ckpt-root=*)
            CKPT_ROOT="${1#*=}"
            shift
            ;;
        --)
            shift
            break
            ;;
        *)
            break
            ;;
    esac
done

FIND_ARGS=("$SWEEP_ID")
if [ -n "$CKPT_ROOT" ]; then
    FIND_ARGS+=(--ckpt-root "$CKPT_ROOT")
fi
for filter in "${RUN_FILTERS[@]}"; do
    FIND_ARGS+=(--filter "$filter")
done

CKPTS=$(python -m scripts.find_sweep_checkpoints "${FIND_ARGS[@]}")

if [ -z "$CKPTS" ]; then
    if [ ${#RUN_FILTERS[@]} -gt 0 ]; then
        echo "No checkpoints found for sweep_id=$SWEEP_ID with filters: ${RUN_FILTERS[*]}"
    else
        echo "No checkpoints found for sweep_id=$SWEEP_ID"
    fi
    exit 1
fi

# Join ckpts with commas for Hydra multirun
CKPTS_CSV=$(echo "$CKPTS" | paste -sd ',')

echo "Found checkpoints: $CKPTS_CSV"
python eval.py -m "ckpt=$CKPTS_CSV" "$@"
