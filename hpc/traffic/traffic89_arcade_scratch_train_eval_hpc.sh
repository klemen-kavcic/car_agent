#!/bin/bash
# Matched control experiment: arcade reverse controls, ordinary random 10-car
# traffic from the first step (no recovery curriculum), five 40M seeds.
# Final held-out evaluation/trajectories run in each training allocation.
# Submit through submit_traffic89_reverse_controls_pipeline.sh.
#SBATCH --job-name=traffic89_arcade
#SBATCH --partition=all
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --array=0-4%6
#SBATCH --output=/d/hpc/home/kk42117/mag/car1/logs/traffic89_arcade_%A_%a.out
#SBATCH --error=/d/hpc/home/kk42117/mag/car1/logs/traffic89_arcade_%A_%a.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1

ROOT=/d/hpc/home/kk42117/mag/car1
SIF="$ROOT/mlagents.sif"
BUILD="$ROOT/Linux_traffic_training_vehicle_arcade/car.x86_64"
INFO="$ROOT/Linux_traffic_training_vehicle_arcade/build_info.txt"
EVAL_BUILD="$ROOT/Linux_traffic_all_validation_vehicle_arcade/car.x86_64"
EVAL_INFO="$ROOT/Linux_traffic_all_validation_vehicle_arcade/build_info.txt"
PREPARE="$ROOT/prepare_multiagent_config.py"
POST_EVAL="$ROOT/traffic_vehicle_distance_post_train_eval.sh"
SELECTOR="$ROOT/select_multiagent_checkpoint.py"
STATS_PATCH="$ROOT/stats_patched.py"
TRAINER_PATCH="$ROOT/trainer_controller_patched.py"
SEED=${SLURM_ARRAY_TASK_ID:?array task required}
TRAIN_CARS=10; NUM_ENVS=6; STEPS=40000000
SOURCE=${TRAFFIC_TEMPLATE_CONFIG:-"$ROOT/templates/traffic89_laser12_cap3_template.yaml"}
RUN_ID="traffic89_arcade_scratch_${SLURM_ARRAY_JOB_ID}_s${SEED}"
RUN_DIR="$ROOT/results/$RUN_ID"; CONFIG="$ROOT/configs/${RUN_ID}.yaml"

for f in "$SIF" "$BUILD" "$INFO" "$EVAL_BUILD" "$EVAL_INFO" "$PREPARE" "$POST_EVAL" "$SELECTOR" "$STATS_PATCH" "$TRAINER_PATCH" "$SOURCE"; do
    [[ -s "$f" ]] || { echo "ERROR: missing or empty $f" >&2; exit 1; }
done
for line in 'Traffic car pool size: 10;' 'Traffic map source: VoronoiTrain' \
            'Configured Vector Observation Size: 89  [MATCH]' 'Actions: 2 continuous, discrete branches [2]'; do
    grep -Fq "$line" "$INFO" || { echo "ERROR: arcade training build lacks '$line'" >&2; exit 1; }
    grep -Fq "$line" "$EVAL_INFO" || { echo "ERROR: arcade validation build lacks '$line'" >&2; exit 1; }
done
mkdir -p "$ROOT/configs" "$ROOT/results" "$ROOT/logs"

apptainer exec "$SIF" python3 "$PREPARE" --source "$SOURCE" --out "$CONFIG" \
    --steps "$STEPS" --num-envs "$NUM_ENVS" --summary-freq 100000 \
    --vehicle-distance --arcade-reverse-controls

# A task receives its own locked 64-port block, avoiding controller/worker
# startup collisions while six Unity environments are active.
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

echo "Training $RUN_ID: arcade controls, ordinary 10-car traffic, 89 obs, ${NUM_ENVS} environments, ${STEPS} steps, port $PORT_BASE"
apptainer exec \
    --bind "$STATS_PATCH:/usr/local/lib/python3.10/dist-packages/mlagents/trainers/stats.py" \
    --bind "$TRAINER_PATCH:/usr/local/lib/python3.10/dist-packages/mlagents/trainers/trainer_controller.py" \
    "$SIF" xvfb-run -a mlagents-learn "$CONFIG" --env="$BUILD" --run-id="$RUN_ID" \
        --results-dir="$ROOT/results" --no-graphics --torch-device=cpu --num-envs="$NUM_ENVS" \
        --base-port="$PORT_BASE" --seed="$SEED" --env-args -job-worker-count 1 -traffic-car-count "$TRAIN_CARS"
test -s "$RUN_DIR/CarAgent.onnx"
python3 "$SELECTOR" --run-dir "$RUN_DIR" --max-steps "$STEPS" --validate-all
export ROOT SIF PORT_BASE TRAFFIC_EVAL_BUILD="$EVAL_BUILD" TRAFFIC_EVAL_BUILD_INFO="$EVAL_INFO"
bash "$POST_EVAL" "$RUN_ID" "$RUN_DIR/CarAgent.onnx" "$CONFIG"
echo "Done: $RUN_ID"
