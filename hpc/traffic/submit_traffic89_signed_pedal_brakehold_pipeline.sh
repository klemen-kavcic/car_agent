#!/bin/bash
# Submit the revised signed-pedal, brake-before-reverse experiment:
# five independent 40M scratch seeds followed by 5M checkpoint evaluation.
set -euo pipefail

# Run from either mag/car or mag/car1.  Everything a job writes stays in this
# same workspace, so an archive/cleanup in the other workspace cannot overlap.
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT"
mkdir -p "$ROOT/logs"
TRAIN="$ROOT/traffic89_signed_pedal_train_hpc.sh"
EVAL="$ROOT/traffic89_signed_pedal_brakehold_5m_eval_hpc.sh"
for f in "$TRAIN" "$EVAL"; do
    [[ -s "$f" ]] || { echo "ERROR: missing or empty $f" >&2; exit 1; }
done

TRAIN_JOB=$(sbatch --parsable --export='ALL,TRAFFIC_SIGNED_VARIANT=scratch' "$TRAIN")
EVAL_JOB=$(sbatch --parsable --dependency="afterok:${TRAIN_JOB}" \
    --export="ALL,TRAFFIC_TRAIN_JOB_ID=${TRAIN_JOB}" "$EVAL")

echo "Signed-pedal brake-before-reverse training: $TRAIN_JOB (five seeds, 40M each, six environments/seed)"
echo "5M checkpoint evaluation:                 $EVAL_JOB (afterok; 5--40M, 2/5/10 cars, deterministic + stochastic)"
