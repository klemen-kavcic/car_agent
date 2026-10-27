#!/bin/bash
# Submit both new five-seed experiments and automatically start their full
# 1M-checkpoint curves only once each training array has completed successfully.
# Submit: bash submit_traffic89_new_experiments_pipeline.sh
set -euo pipefail

GOAL_RAW=$(sbatch --parsable traffic89_goal_removal_train_eval_hpc.sh)
GOAL_JOB=${GOAL_RAW%%;*}
GOAL_CURVE_RAW=$(sbatch --parsable --dependency="afterok:${GOAL_JOB}" \
    --export="ALL,TRAFFIC_TRAIN_JOB_ID=${GOAL_JOB},TRAFFIC_CURVE_FAMILY=goal-removal" \
    traffic89_checkpoint_curve_after_train_hpc.sh)
GOAL_CURVE_JOB=${GOAL_CURVE_RAW%%;*}

CURRICULUM_RAW=$(sbatch --parsable traffic89_2_5_10_curriculum_train_eval_hpc.sh)
CURRICULUM_JOB=${CURRICULUM_RAW%%;*}
CURRICULUM_CURVE_RAW=$(sbatch --parsable --dependency="afterok:${CURRICULUM_JOB}" \
    --export="ALL,TRAFFIC_TRAIN_JOB_ID=${CURRICULUM_JOB},TRAFFIC_CURVE_FAMILY=2to5to10" \
    traffic89_checkpoint_curve_after_train_hpc.sh)
CURRICULUM_CURVE_JOB=${CURRICULUM_CURVE_RAW%%;*}

echo "Goal-removal training array:       $GOAL_JOB"
echo "Goal-removal 1M evaluation array: $GOAL_CURVE_JOB (afterok)"
echo "2->5->10 training array:          $CURRICULUM_JOB"
echo "2->5->10 1M evaluation array:     $CURRICULUM_CURVE_JOB (afterok)"
