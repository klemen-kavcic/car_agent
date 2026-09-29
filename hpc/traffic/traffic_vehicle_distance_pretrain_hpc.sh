#!/bin/bash
# Vehicle-distance lasers (89 observations): five one-car seeds, 20M steps.
# Submit: sbatch traffic_vehicle_distance_pretrain_hpc.sh
# The resulting checkpoints are used by traffic_vehicle_distance_continue_hpc.sh.
#SBATCH --job-name=traffic89_pre
#SBATCH --partition=all
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
# Includes final multi-car evaluation and trajectories after the 20M pretrain.
#SBATCH --time=72:00:00
#SBATCH --signal=B:TERM@60
#SBATCH --array=0-4%4
#SBATCH --output=/d/hpc/home/kk42117/mag/car1/logs/traffic89_pre_%A_%a.out
#SBATCH --error=/d/hpc/home/kk42117/mag/car1/logs/traffic89_pre_%A_%a.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
ROOT=/d/hpc/home/kk42117/mag/car1
SIF="$ROOT/mlagents.sif"; BUILD="$ROOT/Linux_traffic_training_vehicle/car.x86_64"
INFO="$ROOT/Linux_traffic_training_vehicle/build_info.txt"
SOURCE="$ROOT/results/car_vgrid_19016936_laser12_cap3_s0/configuration.yaml"
PREPARE="$ROOT/prepare_multiagent_config.py"; SEED=${SLURM_ARRAY_TASK_ID:?array task required}
POST_EVAL="$ROOT/traffic_vehicle_distance_post_train_eval.sh"
RUN_ID="traffic89_${SLURM_ARRAY_JOB_ID}_pretrain_s${SEED}"; CONFIG="$ROOT/configs/${RUN_ID}.yaml"
for f in "$SIF" "$BUILD" "$INFO" "$SOURCE" "$PREPARE" "$POST_EVAL"; do [[ -s "$f" ]] || { echo "ERROR: missing $f" >&2; exit 1; }; done
grep -Fq 'Configured Vector Observation Size: 89  [MATCH]' "$INFO" || { echo 'ERROR: build is not the 89-observation vehicle build' >&2; exit 1; }
grep -Fq 'Traffic car pool size: 10;' "$INFO" || { echo 'ERROR: build must contain the ten-car pool' >&2; exit 1; }
mkdir -p "$ROOT/logs" "$ROOT/configs" "$ROOT/results"
python3 "$PREPARE" --source "$SOURCE" --out "$CONFIG" --steps 20000000 --num-envs 64 --vehicle-distance
PORT_BLOCK=64; PORT_FIRST=10000; PORT_SLOT_COUNT=800; PORT_LOCK=
PREFERRED_SLOT=$(((SLURM_ARRAY_JOB_ID * 31 + SEED) % PORT_SLOT_COUNT))
for ((a=0; a<PORT_SLOT_COUNT; a++)); do slot=$(((PREFERRED_SLOT+a)%PORT_SLOT_COUNT)); lock="/tmp/car_mlagents_portblock_${slot}.lock"; if mkdir "$lock" 2>/dev/null; then base=$((PORT_FIRST+slot*PORT_BLOCK)); free=1; if command -v ss >/dev/null 2>&1; then for ((offset=0; offset<PORT_BLOCK; offset++)); do if ss -H -ltn "sport = :$((base+offset))" | grep -q .; then free=0; break; fi; done; fi; if ((free)); then PORT_LOCK="$lock"; PORT_BASE="$base"; break; fi; rmdir "$lock" 2>/dev/null || true; fi; done
[[ -n "$PORT_LOCK" ]] || { echo 'ERROR: no free port block' >&2; exit 1; }
cleanup(){ rmdir "$PORT_LOCK" 2>/dev/null || true; }; trap cleanup EXIT; trap 'cleanup; exit 143' INT TERM
echo "Training $RUN_ID: one car, 64 envs, 20M steps, seed=$SEED, obs=89"
apptainer exec "$SIF" xvfb-run -a mlagents-learn "$CONFIG" --env="$BUILD" --run-id="$RUN_ID" --results-dir="$ROOT/results" --no-graphics --torch-device=cpu --num-envs=64 --base-port="$PORT_BASE" --seed="$SEED" --env-args -job-worker-count 1 -traffic-car-count 1
test -s "$ROOT/results/$RUN_ID/CarAgent.onnx"
bash "$POST_EVAL" "$RUN_ID" "$ROOT/results/$RUN_ID/CarAgent.onnx" "$CONFIG"
