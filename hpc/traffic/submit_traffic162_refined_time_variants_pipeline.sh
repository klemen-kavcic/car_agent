#!/bin/bash
# Submit 2 variants × 5 seeds, then their 5M checkpoint evaluations.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TRAIN="$ROOT/traffic162_refined_time_variants_train_hpc.sh"
EVAL="$ROOT/traffic162_refined_time_variants_5m_eval_hpc.sh"
for f in "$TRAIN" "$EVAL"; do [[ -s "$f" ]] || { echo "ERROR: missing or empty $f" >&2; exit 1; }; done
mkdir -p "$ROOT/logs"; cd "$ROOT"
TRAIN_JOB=$(sbatch --parsable "$TRAIN")
EVAL_JOB=$(sbatch --parsable --dependency="afterok:${TRAIN_JOB}" \
    --export="ALL,TRAFFIC_TRAIN_JOB_ID=${TRAIN_JOB}" "$EVAL")
echo "Refined 162-observation training: $TRAIN_JOB (10 tasks: 2 variants × 5 seeds)"
echo "5M checkpoint evaluation:          $EVAL_JOB (afterok; 5--40M, 2/5/10 cars, deterministic/stochastic)"
echo "Variants: time_remove_goal; time_remove_goal_gravel"
