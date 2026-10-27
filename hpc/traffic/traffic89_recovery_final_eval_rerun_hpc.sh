#!/bin/bash
# Re-run only the final evaluation/trajectories for a completed recovery
# training array whose post-training helper failed before ROOT/SIF were exported.
# Example:
#   sbatch --export=ALL,TRAFFIC_TRAIN_JOB_ID=20155935 traffic89_recovery_final_eval_rerun_hpc.sh
#SBATCH --job-name=traffic89_reval
#SBATCH --partition=all
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --array=0-4%5
#SBATCH --output=/d/hpc/home/kk42117/mag/car1/logs/traffic89_reval_%A_%a.out
#SBATCH --error=/d/hpc/home/kk42117/mag/car1/logs/traffic89_reval_%A_%a.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1

ROOT=/d/hpc/home/kk42117/mag/car1
SIF="$ROOT/mlagents.sif"
POST_EVAL="$ROOT/traffic_vehicle_distance_post_train_eval.sh"
TRAIN_JOB=${TRAFFIC_TRAIN_JOB_ID:?provide TRAFFIC_TRAIN_JOB_ID, e.g. 20155935}
SEED=${SLURM_ARRAY_TASK_ID:?array task required}
RUN_ID="traffic89_recovery_${TRAIN_JOB}_s${SEED}"
RUN_DIR="$ROOT/results/$RUN_ID"
MODEL="$RUN_DIR/CarAgent.onnx"
CONFIG="$RUN_DIR/configuration.yaml"

for file in "$SIF" "$POST_EVAL" "$MODEL" "$CONFIG"; do
    [[ -s "$file" ]] || { echo "ERROR: missing or empty $file" >&2; exit 1; }
done

# Reserve exactly one 64-port range per concurrent evaluator, matching the
# guarded training jobs and avoiding the previous worker/port collisions.
PORT_BLOCK=64; PORT_FIRST=10000; PORT_SLOT_COUNT=800; PORT_LOCK=
PREFERRED_SLOT=$(((SLURM_ARRAY_JOB_ID * 31 + SEED) % PORT_SLOT_COUNT))
for ((ATTEMPT=0; ATTEMPT<PORT_SLOT_COUNT; ATTEMPT++)); do
    SLOT=$(((PREFERRED_SLOT + ATTEMPT) % PORT_SLOT_COUNT))
    CANDIDATE_LOCK="/tmp/car_mlagents_portblock_${SLOT}.lock"
    if mkdir "$CANDIDATE_LOCK" 2>/dev/null; then
        CANDIDATE_BASE=$((PORT_FIRST + SLOT * PORT_BLOCK)); BLOCK_FREE=1
        if command -v ss >/dev/null 2>&1; then
            for ((OFFSET=0; OFFSET<PORT_BLOCK; OFFSET++)); do
                if ss -H -ltn "sport = :$((CANDIDATE_BASE + OFFSET))" | grep -q .; then BLOCK_FREE=0; break; fi
            done
        fi
        if ((BLOCK_FREE)); then PORT_LOCK="$CANDIDATE_LOCK"; PORT_BASE="$CANDIDATE_BASE"; break; fi
        rmdir "$CANDIDATE_LOCK" 2>/dev/null || true
    fi
done
[[ -n "$PORT_LOCK" ]] || { echo 'ERROR: no free ML-Agents port block' >&2; exit 1; }
cleanup(){ rmdir "$PORT_LOCK" 2>/dev/null || true; }
trap cleanup EXIT; trap 'cleanup; exit 143' INT TERM

echo "Final evaluation rerun: $RUN_ID, port $PORT_BASE"
export ROOT SIF PORT_BASE
bash "$POST_EVAL" "$RUN_ID" "$MODEL" "$CONFIG"
echo "Done: $RUN_ID"
