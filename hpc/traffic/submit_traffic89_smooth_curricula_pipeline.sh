#!/bin/bash
# Submit the two smooth traffic curricula and their post-training 1M checkpoint
# curves. The two curve arrays wait for the combined training array, then each
# evaluate its own five seed directories in parallel (2/5/10 cars x det/stoch
# within every array task).
set -euo pipefail

ROOT=/d/hpc/home/kk42117/mag/car1
TRAIN="$ROOT/traffic89_smooth_curricula_train_eval_hpc.sh"
CURVES="$ROOT/traffic89_checkpoint_curve_after_train_hpc.sh"

for file in "$TRAIN" "$CURVES"; do
    [[ -s "$file" ]] || { echo "ERROR: missing $file" >&2; exit 1; }
done

TRAIN_JOB=$(sbatch --parsable "$TRAIN")
COUNT_CURVES=$(sbatch --parsable --dependency="afterok:${TRAIN_JOB}" \
    --export="ALL,TRAFFIC_TRAIN_JOB_ID=${TRAIN_JOB},TRAFFIC_CURVE_FAMILY=smooth-count" "$CURVES")
ROUTE_CURVES=$(sbatch --parsable --dependency="afterok:${TRAIN_JOB}" \
    --export="ALL,TRAFFIC_TRAIN_JOB_ID=${TRAIN_JOB},TRAFFIC_CURVE_FAMILY=smooth-route" "$CURVES")

echo "Training:               $TRAIN_JOB (tasks 0-4 count, 5-9 route)"
echo "Count 1M curve eval:    $COUNT_CURVES (afterok)"
echo "Route 1M curve eval:    $ROUTE_CURVES (afterok)"
