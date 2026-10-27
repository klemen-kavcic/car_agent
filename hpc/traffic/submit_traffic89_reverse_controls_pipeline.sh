#!/bin/bash
# Launch three matched five-seed 40M experiments and their dependent 1M
# checkpoint evaluations:
#   1. explicit gear + recovery curriculum;
#   2. arcade controls + the same recovery curriculum;
#   3. arcade controls + ordinary 10-car traffic from step 0.
set -euo pipefail

ROOT=/d/hpc/home/kk42117/mag/car1
RECOVERY="$ROOT/traffic89_recovery_curriculum_train_eval_hpc.sh"
ARCADE_SCRATCH="$ROOT/traffic89_arcade_scratch_train_eval_hpc.sh"
CURVES="$ROOT/traffic89_checkpoint_curve_after_train_hpc.sh"
for file in "$RECOVERY" "$ARCADE_SCRATCH" "$CURVES"; do
    [[ -s "$file" ]] || { echo "ERROR: missing $file" >&2; exit 1; }
done

EXPLICIT_JOB=$(sbatch --parsable --export='ALL,TRAFFIC_CONTROL_MODE=explicit,TRAFFIC_RECOVERY_VARIANT=unshaped' "$RECOVERY")
ARCADE_RECOVERY_JOB=$(sbatch --parsable --export='ALL,TRAFFIC_CONTROL_MODE=arcade,TRAFFIC_RECOVERY_VARIANT=unshaped' "$RECOVERY")
ARCADE_SCRATCH_JOB=$(sbatch --parsable "$ARCADE_SCRATCH")

EXPLICIT_CURVES=$(sbatch --parsable --dependency="afterok:${EXPLICIT_JOB}" \
    --export="ALL,TRAFFIC_TRAIN_JOB_ID=${EXPLICIT_JOB},TRAFFIC_CURVE_FAMILY=recovery-unshaped" "$CURVES")
ARCADE_RECOVERY_CURVES=$(sbatch --parsable --dependency="afterok:${ARCADE_RECOVERY_JOB}" \
    --export="ALL,TRAFFIC_TRAIN_JOB_ID=${ARCADE_RECOVERY_JOB},TRAFFIC_CURVE_FAMILY=recovery-arcade" "$CURVES")
ARCADE_SCRATCH_CURVES=$(sbatch --parsable --dependency="afterok:${ARCADE_SCRATCH_JOB}" \
    --export="ALL,TRAFFIC_TRAIN_JOB_ID=${ARCADE_SCRATCH_JOB},TRAFFIC_CURVE_FAMILY=arcade-scratch" "$CURVES")

echo "Explicit recovery training:        $EXPLICIT_JOB"
echo "  1M checkpoint evaluation:       $EXPLICIT_CURVES (afterok)"
echo "Arcade recovery training:          $ARCADE_RECOVERY_JOB"
echo "  1M checkpoint evaluation:       $ARCADE_RECOVERY_CURVES (afterok)"
echo "Arcade ordinary 40M training:     $ARCADE_SCRATCH_JOB"
echo "  1M checkpoint evaluation:       $ARCADE_SCRATCH_CURVES (afterok)"
