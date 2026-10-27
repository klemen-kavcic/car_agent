#!/bin/bash
# Submit two five-seed signed-torque 40M experiments and final-only dependent
# evaluations. This deliberately does not submit 1M checkpoint-curve jobs.
set -euo pipefail

ROOT=/d/hpc/home/kk42117/mag/car1
TRAIN="$ROOT/traffic89_signed_pedal_train_hpc.sh"
EVAL="$ROOT/traffic89_signed_pedal_final_eval_hpc.sh"
for f in "$TRAIN" "$EVAL"; do
    [[ -s "$f" ]] || { echo "ERROR: missing $f" >&2; exit 1; }
done

CURRICULUM_TRAIN=$(sbatch --parsable --export='ALL,TRAFFIC_SIGNED_VARIANT=curriculum' "$TRAIN")
SCRATCH_TRAIN=$(sbatch --parsable --export='ALL,TRAFFIC_SIGNED_VARIANT=scratch' "$TRAIN")
CURRICULUM_EVAL=$(sbatch --parsable --dependency="afterok:${CURRICULUM_TRAIN}" \
    --export="ALL,TRAFFIC_TRAIN_JOB_ID=${CURRICULUM_TRAIN},TRAFFIC_SIGNED_VARIANT=curriculum" "$EVAL")
SCRATCH_EVAL=$(sbatch --parsable --dependency="afterok:${SCRATCH_TRAIN}" \
    --export="ALL,TRAFFIC_TRAIN_JOB_ID=${SCRATCH_TRAIN},TRAFFIC_SIGNED_VARIANT=scratch" "$EVAL")

echo "Signed-torque curriculum training: $CURRICULUM_TRAIN (five seeds, 40M each)"
echo "  final 40M evaluation:           $CURRICULUM_EVAL (afterok)"
echo "Signed-torque ordinary training:   $SCRATCH_TRAIN (five seeds, 40M each)"
echo "  final 40M evaluation:           $SCRATCH_EVAL (afterok)"
