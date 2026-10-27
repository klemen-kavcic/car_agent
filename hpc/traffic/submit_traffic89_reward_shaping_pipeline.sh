#!/bin/bash
# Submit the two reward-only 40M experiments. Final 40M evaluation runs first;
# the optional eight-checkpoint (5M interval) sweep waits for it to succeed.
set -euo pipefail
ROOT=/d/hpc/home/kk42117/mag/car1
TRAIN="$ROOT/traffic89_reward_shaping_train_hpc.sh"; FINAL="$ROOT/traffic89_reward_shaping_final_eval_hpc.sh"; CURVES="$ROOT/traffic89_reward_shaping_5m_eval_hpc.sh"
for f in "$TRAIN" "$FINAL" "$CURVES"; do [[ -s "$f" ]] || { echo "ERROR: missing $f" >&2; exit 1; }; done
IDLE_TRAIN=$(sbatch --parsable --export='ALL,TRAFFIC_REWARD_VARIANT=idle' "$TRAIN")
EXP_TRAIN=$(sbatch --parsable --export='ALL,TRAFFIC_REWARD_VARIANT=exponential' "$TRAIN")
IDLE_FINAL=$(sbatch --parsable --dependency="afterok:${IDLE_TRAIN}" --export="ALL,TRAFFIC_TRAIN_JOB_ID=${IDLE_TRAIN},TRAFFIC_REWARD_VARIANT=idle" "$FINAL")
EXP_FINAL=$(sbatch --parsable --dependency="afterok:${EXP_TRAIN}" --export="ALL,TRAFFIC_TRAIN_JOB_ID=${EXP_TRAIN},TRAFFIC_REWARD_VARIANT=exponential" "$FINAL")
IDLE_CURVES=$(sbatch --parsable --dependency="afterok:${IDLE_FINAL}" --export="ALL,TRAFFIC_TRAIN_JOB_ID=${IDLE_TRAIN},TRAFFIC_REWARD_VARIANT=idle" "$CURVES")
EXP_CURVES=$(sbatch --parsable --dependency="afterok:${EXP_FINAL}" --export="ALL,TRAFFIC_TRAIN_JOB_ID=${EXP_TRAIN},TRAFFIC_REWARD_VARIANT=exponential" "$CURVES")
echo "Idle training:                $IDLE_TRAIN"
echo "  final 40M evaluation:      $IDLE_FINAL (afterok)"
echo "  5M checkpoint evaluations: $IDLE_CURVES (after final evaluation)"
echo "Exponential training:         $EXP_TRAIN"
echo "  final 40M evaluation:      $EXP_FINAL (afterok)"
echo "  5M checkpoint evaluations: $EXP_CURVES (after final evaluation)"
