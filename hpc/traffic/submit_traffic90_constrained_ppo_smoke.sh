#!/bin/bash
# One 1M-step seed that proves the CPO dual update/logging path.
# It intentionally assigns every completed car cost=1, target=0, so lambda
# must grow. Do not evaluate or compare this deliberately invalid policy.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TRAIN="$ROOT/traffic_limits_variants_train_hpc.sh"
[[ -s "$TRAIN" ]] || { echo "ERROR: missing $TRAIN" >&2; exit 1; }

JOB=$(sbatch --parsable --array=0 --export=ALL,TRAFFIC_LIMITS_VARIANT=time90_limits_cpo_smoke,TRAFFIC_LIMITS_STEPS=1000000 "$TRAIN")
echo "CPO forced-cost smoke test: training=$JOB (one seed, 1M steps; no evaluation)"
