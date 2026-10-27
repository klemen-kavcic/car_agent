#!/bin/bash
# Five-seed, matched 162-observation PPO middle-ground experiment.
# Compared with the original 162/256 run, only PPO hyperparameters change:
# LR 0.0002, buffer 16384, batch 2048.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TRAIN="$ROOT/traffic_limits_variants_train_hpc.sh"
EVAL="$ROOT/traffic_limits_variants_5m_eval_hpc.sh"
for f in "$TRAIN" "$EVAL"; do
    [[ -s "$f" ]] || { echo "ERROR: missing or empty $f" >&2; exit 1; }
done

mkdir -p "$ROOT/logs"
cd "$ROOT"
VARIANT=dense162_limits_256_middle
TRAIN_JOB=$(sbatch --parsable --export="ALL,TRAFFIC_LIMITS_VARIANT=${VARIANT}" "$TRAIN")
EVAL_JOB=$(sbatch --parsable --dependency="afterok:${TRAIN_JOB}" \
    --export="ALL,TRAFFIC_LIMITS_VARIANT=${VARIANT},TRAFFIC_LIMITS_TRAIN_JOB_ID=${TRAIN_JOB}" "$EVAL")
echo "${VARIANT}: training=${TRAIN_JOB}; 5M evaluation=${EVAL_JOB} (afterok)"
