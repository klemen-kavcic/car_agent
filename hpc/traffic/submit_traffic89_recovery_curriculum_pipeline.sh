#!/bin/bash
# Submit the five recovery-curriculum training seeds and, only after every
# training task succeeds, the independent per-1M checkpoint evaluation array.
set -euo pipefail

ROOT=/d/hpc/home/kk42117/mag/car1
TRAIN="$ROOT/traffic89_recovery_curriculum_train_eval_hpc.sh"
CURVES="$ROOT/traffic89_checkpoint_curve_after_train_hpc.sh"
VARIANT=${TRAFFIC_RECOVERY_VARIANT:-unshaped}
[[ "$VARIANT" =~ ^(unshaped|clearance)$ ]] || {
    echo "ERROR: TRAFFIC_RECOVERY_VARIANT must be unshaped or clearance" >&2; exit 1;
}
for file in "$TRAIN" "$CURVES"; do
    [[ -s "$file" ]] || { echo "ERROR: missing $file" >&2; exit 1; }
done

TRAIN_JOB=$(sbatch --parsable --export="ALL,TRAFFIC_RECOVERY_VARIANT=${VARIANT}" "$TRAIN")
CURVE_JOB=$(sbatch --parsable --dependency="afterok:${TRAIN_JOB}" \
    --export="ALL,TRAFFIC_TRAIN_JOB_ID=${TRAIN_JOB},TRAFFIC_CURVE_FAMILY=recovery-${VARIANT}" "$CURVES")
echo "Recovery ${VARIANT} training: $TRAIN_JOB (five seeds; 13M recovery then normal 10-car traffic)"
echo "1M checkpoint evaluation: $CURVE_JOB (afterok; 2/5/10 cars × deterministic/stochastic)"
