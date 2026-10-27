#!/bin/bash
# Five-seed safe-learning test: 90 observations with an adaptive collision
# penalty targeting <=8% VehicleCollision. All other settings match the
# successful time90_limits_128 remove-finished-cars run.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TRAIN="$ROOT/traffic_limits_variants_train_hpc.sh"
EVAL="$ROOT/traffic_limits_variants_5m_eval_hpc.sh"
for f in "$TRAIN" "$EVAL"; do
    [[ -s "$f" ]] || { echo "ERROR: missing or empty $f" >&2; exit 1; }
done

mkdir -p "$ROOT/logs"
cd "$ROOT"
VARIANT=time90_limits_lagrangian
TRAIN_JOB=$(sbatch --parsable --export="ALL,TRAFFIC_LIMITS_VARIANT=${VARIANT}" "$TRAIN")
EVAL_JOB=$(sbatch --parsable --dependency="afterok:${TRAIN_JOB}" \
    --export="ALL,TRAFFIC_LIMITS_VARIANT=${VARIANT},TRAFFIC_LIMITS_TRAIN_JOB_ID=${TRAIN_JOB}" "$EVAL")
echo "${VARIANT}: training=${TRAIN_JOB}; 5M evaluation=${EVAL_JOB} (afterok)"
