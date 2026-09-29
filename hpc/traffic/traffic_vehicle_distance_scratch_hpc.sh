#!/bin/bash
# 89-observation vehicle-distance lasers: 2/5/10 cars, five seeds, 40M steps.
# Submit from car1: sbatch traffic_vehicle_distance_scratch_hpc.sh
# This is a separate experiment; it must not share run IDs with the 77-value sweep.
#SBATCH --job-name=traffic89_scratch
#SBATCH --partition=all
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
# Training plus final 250-episode deterministic/stochastic evaluation at 2/5/10
# cars and trajectory collection run in this same task.
#SBATCH --time=72:00:00
#SBATCH --signal=B:TERM@60
#SBATCH --array=0-14%4
#SBATCH --output=/d/hpc/home/kk42117/mag/car1/logs/traffic89_scratch_%A_%a.out
#SBATCH --error=/d/hpc/home/kk42117/mag/car1/logs/traffic89_scratch_%A_%a.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
ROOT=/d/hpc/home/kk42117/mag/car1
SIF="$ROOT/mlagents.sif"
BUILD="$ROOT/Linux_traffic_training_vehicle/car.x86_64"
BUILD_INFO="$ROOT/Linux_traffic_training_vehicle/build_info.txt"
SOURCE="$ROOT/results/car_vgrid_19016936_laser12_cap3_s0/configuration.yaml"
PREPARE="$ROOT/prepare_multiagent_config.py"
POST_EVAL="$ROOT/traffic_vehicle_distance_post_train_eval.sh"
TASK=${SLURM_ARRAY_TASK_ID:?array task required}
if ((TASK < 0 || TASK > 14)); then echo 'ERROR: array index must be 0..14' >&2; exit 1; fi
DENSITIES=(2 5 10); CARS=${DENSITIES[$((TASK / 5))]}; SEED=$((TASK % 5))
ENVS=(32 12 6); NUM_ENVS=${ENVS[$((TASK / 5))]}
RUN_ID="traffic89_${SLURM_ARRAY_JOB_ID}_scratch_cars${CARS}_s${SEED}"
CONFIG="$ROOT/configs/${RUN_ID}.yaml"; RESULTS="$ROOT/results/$RUN_ID"
for required in "$SIF" "$BUILD" "$BUILD_INFO" "$SOURCE" "$PREPARE" "$POST_EVAL"; do
  [[ -s "$required" ]] || { echo "ERROR: missing or empty $required" >&2; exit 1; }
done
for line in 'Traffic car pool size: 10;' 'Configured Vector Observation Size: 89  [MATCH]'; do
  grep -Fq "$line" "$BUILD_INFO" || { echo "ERROR: build lacks '$line'" >&2; exit 1; }
done
mkdir -p "$ROOT/logs" "$ROOT/configs" "$RESULTS"
python3 "$PREPARE" --source "$SOURCE" --out "$CONFIG" --steps 40000000 --num-envs "$NUM_ENVS" --vehicle-distance
PORT_BLOCK=64; PORT_FIRST=10000; PORT_SLOT_COUNT=800; PORT_LOCK=; PREFERRED_SLOT=$(((SLURM_ARRAY_JOB_ID * 31 + TASK) % PORT_SLOT_COUNT))
for ((ATTEMPT=0; ATTEMPT<PORT_SLOT_COUNT; ATTEMPT++)); do
  SLOT=$(((PREFERRED_SLOT + ATTEMPT) % PORT_SLOT_COUNT)); CANDIDATE_LOCK="/tmp/car_mlagents_portblock_${SLOT}.lock"
  if mkdir "$CANDIDATE_LOCK" 2>/dev/null; then CANDIDATE_BASE=$((PORT_FIRST + SLOT * PORT_BLOCK)); BLOCK_FREE=1; if command -v ss >/dev/null 2>&1; then for ((OFFSET=0; OFFSET<PORT_BLOCK; OFFSET++)); do if ss -H -ltn "sport = :$((CANDIDATE_BASE + OFFSET))" | grep -q .; then BLOCK_FREE=0; break; fi; done; fi; if ((BLOCK_FREE)); then PORT_LOCK="$CANDIDATE_LOCK"; PORT_BASE="$CANDIDATE_BASE"; break; fi; rmdir "$CANDIDATE_LOCK" 2>/dev/null || true; fi
done
[[ -n "$PORT_LOCK" ]] || { echo 'ERROR: no free port block' >&2; exit 1; }
cleanup(){ rmdir "$PORT_LOCK" 2>/dev/null || true; }; trap cleanup EXIT; trap 'cleanup; exit 143' INT TERM
echo "Training $RUN_ID: scratch cars=$CARS envs=$NUM_ENVS steps=40000000 obs=89 port=$PORT_BASE"
apptainer exec "$SIF" xvfb-run -a mlagents-learn "$CONFIG" --env="$BUILD" --run-id="$RUN_ID" --results-dir="$ROOT/results" --no-graphics --torch-device=cpu --num-envs="$NUM_ENVS" --base-port="$PORT_BASE" --seed="$SEED" --env-args -job-worker-count 1 -traffic-car-count "$CARS"
test -s "$RESULTS/CarAgent.onnx"
bash "$POST_EVAL" "$RUN_ID" "$RESULTS/CarAgent.onnx" "$CONFIG"
echo "Done: $RESULTS/CarAgent.onnx"
