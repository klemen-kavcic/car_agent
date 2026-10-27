#!/bin/bash
# One 1M-step, unforced terminal-marker test.
# Target=0 means any real VehicleCollision must make lambda rise. Unlike the
# forced-cost smoke run, this validates the Unity terminal marker -> trainer
# data path and produces no meaningful policy for evaluation.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TRAIN="$ROOT/traffic_limits_variants_train_hpc.sh"
[[ -s "$TRAIN" ]] || { echo "ERROR: missing $TRAIN" >&2; exit 1; }

JOB=$(sbatch --parsable --array=0 --export=ALL,TRAFFIC_LIMITS_VARIANT=time90_limits_cpo_marker_smoke,TRAFFIC_LIMITS_STEPS=1000000 "$TRAIN")
echo "CPO real-marker smoke test: training=$JOB (one seed, 1M steps; no evaluation)"
