#!/bin/bash
# Submit shared solo bases, then fine-tunes and their evaluations automatically.
set -euo pipefail
SOLO_JOB=$(sbatch --parsable traffic89_solo20m_hpc.sh)
SOLO_EVAL_JOB=$(sbatch --parsable --dependency=afterok:$SOLO_JOB --array=0-4%3 traffic89_transfer_final_eval_hpc.sh)
TRANSFER_JOB=$(sbatch --parsable --dependency=afterok:$SOLO_JOB traffic89_transfer_finetune20m_hpc.sh)
TRANSFER_EVAL_JOB=$(sbatch --parsable --dependency=afterok:$TRANSFER_JOB --array=5-19%3 traffic89_transfer_final_eval_hpc.sh)
echo "Solo bases: $SOLO_JOB"
echo "Solo final evaluations (after solo success): $SOLO_EVAL_JOB"
echo "Traffic fine-tunes (after solo success): $TRANSFER_JOB"
echo "Fine-tune final evaluations (after traffic success): $TRANSFER_EVAL_JOB"
