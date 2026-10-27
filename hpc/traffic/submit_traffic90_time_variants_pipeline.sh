#!/bin/bash
# Submit 4 variants x 5 seeds (20 training tasks), then the matching 5M
# checkpoint evaluations.  The dependency waits until every training task has
# completed successfully, so evaluation never reads partial checkpoints.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TRAIN="$ROOT/traffic90_time_variants_train_hpc.sh"
EVAL="$ROOT/traffic90_time_variants_5m_eval_hpc.sh"
for f in "$TRAIN" "$EVAL"; do
    [[ -s "$f" ]] || { echo "ERROR: missing $f" >&2; exit 1; }
done
mkdir -p "$ROOT/logs"
cd "$ROOT"

TRAIN_JOB=$(sbatch --parsable "$TRAIN")
EVAL_JOB=$(sbatch --parsable --dependency="afterok:${TRAIN_JOB}" \
    --export="ALL,TRAFFIC_TRAIN_JOB_ID=${TRAIN_JOB}" "$EVAL")

echo "90-observation training array: $TRAIN_JOB (20 tasks: 4 variants x 5 seeds)"
echo "5M checkpoint evaluation array: $EVAL_JOB (afterok; 5--40M, 2/5/10 cars, deterministic/stochastic)"
echo "Variants: time; time_remove_goal; time_remove_goal_limits; time_remove_goal_gravel"
