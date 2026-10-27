#!/bin/bash
# Train five seeds of the normal-map dense-lidar control, then evaluate every 5M.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TRAIN="$ROOT/traffic162_dense_time_goalremoval_train_hpc.sh"
EVAL="$ROOT/traffic162_dense_time_goalremoval_5m_eval_hpc.sh"
for f in "$TRAIN" "$EVAL"; do [[ -s "$f" ]] || { echo "ERROR: missing or empty $f" >&2; exit 1; }; done
mkdir -p "$ROOT/logs"; cd "$ROOT"
TRAIN_JOB=$(sbatch --parsable "$TRAIN")
EVAL_JOB=$(sbatch --parsable --dependency="afterok:${TRAIN_JOB}" \
    --export="ALL,TRAFFIC_DENSE_TRAIN_JOB_ID=${TRAIN_JOB}" "$EVAL")
echo "Normal-map dense-lidar training: $TRAIN_JOB (5 seeds, 40M each)"
echo "5M checkpoint evaluation:        $EVAL_JOB (afterok; 5--40M, 2/5/10 cars, deterministic/stochastic)"
