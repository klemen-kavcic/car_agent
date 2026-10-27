#!/bin/bash
# Eight matched 90-observation experiments. Every entry trains five seeds and
# submits a dependent checkpoint evaluation at every 5M step.
set -euo pipefail

ROOT=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
TRAIN="$ROOT/traffic_limits_variants_train_hpc.sh"
EVAL="$ROOT/traffic_limits_variants_5m_eval_hpc.sh"

[[ -s "$TRAIN" && -s "$EVAL" ]] || { echo 'Missing train/evaluation scripts.' >&2; exit 1; }

VARIANTS=(
  time90_limits_60
  time90_limits_cpo_t10_60
  time90_limits_cpo_t08_60
  time90_limits_cpo_t10_ttc_40
  time90_limits_cpo_t10_ttc_60
  time90_limits_cpo_t08_ttc_40
  time90_limits_cpo_t08_ttc_60
  time90_limits_top5_40
)
STEPS=(60000000 60000000 60000000 40000000 60000000 40000000 60000000 40000000)

for i in "${!VARIANTS[@]}"; do
  variant=${VARIANTS[$i]}
  steps=${STEPS[$i]}
  train_job=$(sbatch --parsable --export="ALL,TRAFFIC_LIMITS_VARIANT=$variant,TRAFFIC_LIMITS_STEPS=$steps" "$TRAIN")
  eval_job=$(sbatch --parsable --dependency="afterok:$train_job" \
    --export="ALL,TRAFFIC_LIMITS_VARIANT=$variant,TRAFFIC_LIMITS_STEPS=$steps,TRAFFIC_LIMITS_TRAIN_JOB_ID=$train_job" "$EVAL")
  echo "$variant: training=$train_job; 5M evaluation=$eval_job (afterok)"
done
