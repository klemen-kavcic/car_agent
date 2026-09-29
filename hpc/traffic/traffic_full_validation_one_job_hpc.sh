#!/bin/bash
# One Slurm allocation for all six traffic conditions. Each child runs one of
# the existing 20 shard/mode tasks, so the sample set matches the array sweep.

#SBATCH --job-name=traffic_val_one
#SBATCH --partition=all
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/d/hpc/home/kk42117/mag/car1/logs/traffic_val_one_%j.out
#SBATCH --error=/d/hpc/home/kk42117/mag/car1/logs/traffic_val_one_%j.err

set -euo pipefail

ROOT=/d/hpc/home/kk42117/mag/car1
SHARD_SCRIPT="$ROOT/traffic_full_validation_hpc.sh"
PARALLEL=${TRAFFIC_PARALLEL:-8}

if [ -z "${SLURM_JOB_ID:-}" ] || [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo 'ERROR: submit this as one ordinary sbatch job, not a job array.' >&2
    exit 1
fi
if ! [[ "$PARALLEL" =~ ^[1-8]$ ]]; then
    echo 'ERROR: TRAFFIC_PARALLEL must be an integer from 1 to 8.' >&2
    exit 1
fi
if [ ! -f "$SHARD_SCRIPT" ]; then
    echo "ERROR: missing $SHARD_SCRIPT" >&2
    exit 1
fi

# A smoke-test setting in the submitting shell must never shorten this sweep.
unset TRAFFIC_SCENARIO TRAFFIC_EPISODES TRAFFIC_ROLLOUTS_PER_MAP

PIDS=()
LABELS=()
stop_children() {
    trap - TERM INT
    for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
    wait 2>/dev/null || true
    exit 143
}
trap stop_children TERM INT

wait_group() {
    local failed=0 i
    for ((i=0; i<${#PIDS[@]}; i++)); do
        if wait "${PIDS[i]}"; then
            echo "PASS ${LABELS[i]}"
        else
            echo "FAIL ${LABELS[i]} (see its log)" >&2
            failed=1
        fi
    done
    PIDS=()
    LABELS=()
    if ((failed)); then exit 1; fi
}

echo "One-job full traffic evaluation: Slurm job $SLURM_JOB_ID"
echo "Six conditions x ten map shards x two modes = 120 evaluations; up to $PARALLEL at once."
echo "Results: $ROOT/traffic_results/full_validation_$SLURM_JOB_ID"

for cars in 2 5 10; do
    for strategy in same mixed; do
        for ((task=0; task<20; task++)); do
            shard=$((task / 2))
            if ((task % 2 == 0)); then mode=deterministic; else mode=stochastic; fi
            label="cars=$cars strategy=$strategy shard=$shard mode=$mode"
            log="$ROOT/logs/traffic_val_one_${SLURM_JOB_ID}_${cars}_${strategy}_${task}.log"
            echo "START $label -> $log"
            env TRAFFIC_CARS="$cars" TRAFFIC_SEED_STRATEGY="$strategy" \
                SLURM_ARRAY_TASK_ID="$task" SLURM_ARRAY_JOB_ID="$SLURM_JOB_ID" \
                bash "$SHARD_SCRIPT" >"$log" 2>&1 &
            PIDS+=("$!")
            LABELS+=("$label")
            if ((${#PIDS[@]} == PARALLEL)); then wait_group; fi
        done
    done
done
if ((${#PIDS[@]})); then wait_group; fi

echo "All 120 evaluations completed: $ROOT/traffic_results/full_validation_$SLURM_JOB_ID"
