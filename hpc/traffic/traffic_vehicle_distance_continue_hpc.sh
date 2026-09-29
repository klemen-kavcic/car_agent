#!/bin/bash
# Initialise from the one-car 20M checkpoints, then train 20M with 2/5/10 cars.
# Submit after pretraining completes: sbatch --export=ALL,PRETRAIN_ARRAY_ID=<jobid> traffic_vehicle_distance_continue_hpc.sh
# Five seeds per car count (15 tasks total).
#SBATCH --job-name=traffic89_cont
#SBATCH --partition=all
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
# Includes final multi-car evaluation and trajectories after the 20M continuation.
#SBATCH --time=72:00:00
#SBATCH --signal=B:TERM@60
#SBATCH --array=0-14%4
#SBATCH --output=/d/hpc/home/kk42117/mag/car1/logs/traffic89_cont_%A_%a.out
#SBATCH --error=/d/hpc/home/kk42117/mag/car1/logs/traffic89_cont_%A_%a.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
ROOT=/d/hpc/home/kk42117/mag/car1; SIF="$ROOT/mlagents.sif"
BUILD="$ROOT/Linux_traffic_training_vehicle/car.x86_64"; INFO="$ROOT/Linux_traffic_training_vehicle/build_info.txt"
PREPARE="$ROOT/prepare_multiagent_config.py"; A=${PRETRAIN_ARRAY_ID:-}
POST_EVAL="$ROOT/traffic_vehicle_distance_post_train_eval.sh"
[[ "$A" =~ ^[0-9]+$ ]] || { echo 'ERROR: set PRETRAIN_ARRAY_ID to the completed pretraining array job ID' >&2; exit 1; }
TASK=${SLURM_ARRAY_TASK_ID:?array task required}; CARS=(2 5 10); NENV=(32 12 6)
GROUP=$((TASK/5)); SEED=$((TASK%5)); NUM_CARS=${CARS[$GROUP]}; NUM_ENVS=${NENV[$GROUP]}
PRE="traffic89_${A}_pretrain_s${SEED}"; SOURCE="$ROOT/results/$PRE"; CHECKPOINT="$SOURCE/CarAgent/checkpoint.pt"
RUN_ID="traffic89_${A}_continue_cars${NUM_CARS}_s${SEED}"; CONFIG="$ROOT/configs/${RUN_ID}.yaml"
for f in "$SIF" "$BUILD" "$INFO" "$PREPARE" "$POST_EVAL" "$SOURCE/configuration.yaml" "$CHECKPOINT"; do [[ -s "$f" ]] || { echo "ERROR: missing $f" >&2; exit 1; }; done
grep -Fq 'Configured Vector Observation Size: 89  [MATCH]' "$INFO" || { echo 'ERROR: wrong build (expected 89 observations)' >&2; exit 1; }
mkdir -p "$ROOT/logs" "$ROOT/configs" "$ROOT/results"
python3 "$PREPARE" --source "$SOURCE/configuration.yaml" --out "$CONFIG" --steps 20000000 --num-envs "$NUM_ENVS" --checkpoint "$CHECKPOINT" --vehicle-distance
PORT_BLOCK=64; PORT_FIRST=10000; PORT_SLOT_COUNT=800; PORT_LOCK=
PREFERRED_SLOT=$(((SLURM_ARRAY_JOB_ID * 31 + TASK) % PORT_SLOT_COUNT))
for ((a=0; a<PORT_SLOT_COUNT; a++)); do slot=$(((PREFERRED_SLOT+a)%PORT_SLOT_COUNT)); lock="/tmp/car_mlagents_portblock_${slot}.lock"; if mkdir "$lock" 2>/dev/null; then base=$((PORT_FIRST+slot*PORT_BLOCK)); free=1; if command -v ss >/dev/null 2>&1; then for ((offset=0; offset<PORT_BLOCK; offset++)); do if ss -H -ltn "sport = :$((base+offset))" | grep -q .; then free=0; break; fi; done; fi; if ((free)); then PORT_LOCK="$lock"; PORT_BASE="$base"; break; fi; rmdir "$lock" 2>/dev/null || true; fi; done
[[ -n "$PORT_LOCK" ]] || { echo 'ERROR: no free port block' >&2; exit 1; }
cleanup(){ rmdir "$PORT_LOCK" 2>/dev/null || true; }; trap cleanup EXIT; trap 'cleanup; exit 143' INT TERM
echo "Training $RUN_ID: $NUM_CARS cars, $NUM_ENVS envs, 20M steps after 20M one-car pretraining, seed=$SEED"
apptainer exec "$SIF" xvfb-run -a mlagents-learn "$CONFIG" --env="$BUILD" --run-id="$RUN_ID" --results-dir="$ROOT/results" --no-graphics --torch-device=cpu --num-envs="$NUM_ENVS" --base-port="$PORT_BASE" --seed="$SEED" --env-args -job-worker-count 1 -traffic-car-count "$NUM_CARS"
test -s "$ROOT/results/$RUN_ID/CarAgent.onnx"
bash "$POST_EVAL" "$RUN_ID" "$ROOT/results/$RUN_ID/CarAgent.onnx" "$CONFIG"
