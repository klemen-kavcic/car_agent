#!/bin/bash
# 2, 5, 10 cars x five cap-3 laser77 fine-tunes and five matched scratch runs.
# Each block of ten tasks uses one car count: five fine-tunes, then five scratch.
# Submit from car1: sbatch traffic_multiagent_train_hpc.sh

#SBATCH --job-name=traffic_train
#SBATCH --partition=all
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@60
#SBATCH --array=0-29%4
#SBATCH --output=/d/hpc/home/kk42117/mag/car1/logs/traffic_train_%A_%a.out
#SBATCH --error=/d/hpc/home/kk42117/mag/car1/logs/traffic_train_%A_%a.err

set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1

ROOT=/d/hpc/home/kk42117/mag/car1
SIF="$ROOT/mlagents.sif"
BUILD="$ROOT/Linux_traffic_training/car.x86_64"
BUILD_INFO="$ROOT/Linux_traffic_training/build_info.txt"
MANIFEST="$ROOT/traffic_training_maps.json"
PREPARE="$ROOT/prepare_multiagent_config.py"
SELECTOR="$ROOT/select_multiagent_checkpoint.py"
RESULTS="$ROOT/results"
STATS_PATCH="$ROOT/stats_patched.py"
TRAINER_PATCH="$ROOT/trainer_controller_patched.py"
TASK=${SLURM_ARRAY_TASK_ID:?Submit as a Slurm array}
if ((TASK < 0 || TASK > 29)); then echo 'ERROR: array index must be 0..29.' >&2; exit 1; fi
DENSITIES=(2 5 10)
ENV_COUNTS=(32 12 6)
BLOCK=$((TASK / 10))
WITHIN=$((TASK % 10))
CARS=${DENSITIES[$BLOCK]}
ENVS=${ENV_COUNTS[$BLOCK]}
if ((WITHIN < 5)); then METHOD=finetune; SEED=$WITHIN; else METHOD=scratch; SEED=$((WITHIN - 5)); fi
if [[ "$METHOD" == finetune ]]; then DEFAULT_STEPS=20000000; else DEFAULT_STEPS=40000000; fi
STEPS=${TRAFFIC_TRAIN_STEPS:-$DEFAULT_STEPS}
if ! [[ "$STEPS" =~ ^[1-9][0-9]*$ ]]; then
    echo 'ERROR: TRAFFIC_TRAIN_STEPS must be positive.' >&2
    exit 1
fi
SOURCE="$RESULTS/car_vgrid_19016936_laser12_cap3_s${SEED}"
SOURCE_CONFIG="$SOURCE/configuration.yaml"
CHECKPOINT="$SOURCE/CarAgent/checkpoint.pt"
RUN_ID="traffic${CARS}_${SLURM_ARRAY_JOB_ID}_${METHOD}_s${SEED}"
RESUME=${TRAFFIC_RESUME:-0}
if ! [[ "$RESUME" =~ ^[01]$ ]]; then echo 'ERROR: TRAFFIC_RESUME must be 0 or 1.' >&2; exit 1; fi
if [[ "$RESUME" == 1 ]]; then
    if ! [[ "${TRAIN_ARRAY_ID:-}" =~ ^[0-9]+$ ]] || [[ -n "${TRAFFIC_TRAIN_STEPS:-}" ]]; then
        echo 'ERROR: resume needs numeric TRAIN_ARRAY_ID and no TRAFFIC_TRAIN_STEPS override.' >&2
        exit 1
    fi
    RUN_ID="traffic${CARS}_${TRAIN_ARRAY_ID}_${METHOD}_s${SEED}"
elif [[ -n "${TRAIN_ARRAY_ID:-}" ]]; then
    echo 'ERROR: TRAIN_ARRAY_ID is for TRAFFIC_RESUME=1 only.' >&2
    exit 1
fi
RUN_DIR="$RESULTS/$RUN_ID"
CONFIG="$ROOT/configs/${RUN_ID}.yaml"

for required in "$SIF" "$BUILD" "$BUILD_INFO" "$MANIFEST" "$PREPARE" "$SELECTOR" \
                "$SOURCE_CONFIG" "$STATS_PATCH" "$TRAINER_PATCH"; do
    if [ ! -s "$required" ]; then echo "ERROR: missing or empty $required" >&2; exit 1; fi
done
if [[ "$METHOD" == finetune && ! -s "$CHECKPOINT" ]]; then
    echo "ERROR: fine-tune needs the original PyTorch checkpoint $CHECKPOINT" >&2
    exit 1
fi
for line in \
    'Traffic map source: VoronoiTrain' \
    'Traffic map selection: random-each-episode' \
    'Traffic scenario manifest: traffic_training_maps' \
    'Traffic placement mode: random-connected-asphalt' \
    'Configured Vector Observation Size: 77  [MATCH]'; do
    if ! grep -Fq "$line" "$BUILD_INFO"; then
        echo "ERROR: training build lacks '$line'. Rebuild only TrafficTraining.unity." >&2
        exit 1
    fi
done
if (( $(grep -Ec '^Traffic seat [0-9]+ behavior: CarAgent' "$BUILD_INFO") != 10 )); then
    echo 'ERROR: all ten pooled cars must use the shared CarAgent behavior.' >&2
    exit 1
fi
if ! grep -Fq 'Traffic car pool size: 10;' "$BUILD_INFO"; then
    echo 'ERROR: expected a ten-car-capable training build.' >&2; exit 1
fi

mkdir -p "$ROOT/logs" "$ROOT/configs" "$RESULTS"
if [[ "$RESUME" == 1 ]]; then
    if [[ ! -s "$CONFIG" || ! -s "$RUN_DIR/CarAgent/checkpoint.pt" ]]; then
        echo "ERROR: no saved config/checkpoint to resume for $RUN_ID." >&2
        exit 1
    fi
else
    PREPARE_ARGS=(--source "$SOURCE_CONFIG" --out "$CONFIG" --steps "$STEPS" --num-envs "$ENVS")
    if [[ "$METHOD" == finetune ]]; then PREPARE_ARGS+=(--checkpoint "$CHECKPOINT"); fi
    apptainer exec "$SIF" python3 "$PREPARE" "${PREPARE_ARGS[@]}"
fi

# Reserve 64 local ports atomically; concurrent jobs on one node cannot collide.
PORT_BLOCK=64
PORT_FIRST=10000
PORT_SLOT_COUNT=800
PORT_LOCK=
PREFERRED_SLOT=$(((SLURM_ARRAY_JOB_ID * 31 + TASK) % PORT_SLOT_COUNT))
for ((ATTEMPT=0; ATTEMPT<PORT_SLOT_COUNT; ATTEMPT++)); do
    SLOT=$(((PREFERRED_SLOT + ATTEMPT) % PORT_SLOT_COUNT))
    CANDIDATE_LOCK="/tmp/car_mlagents_portblock_${SLOT}.lock"
    if mkdir "$CANDIDATE_LOCK" 2>/dev/null; then
        CANDIDATE_BASE=$((PORT_FIRST + SLOT * PORT_BLOCK))
        BLOCK_FREE=1
        if command -v ss >/dev/null 2>&1; then
            for ((OFFSET=0; OFFSET<PORT_BLOCK; OFFSET++)); do
                if ss -H -ltn "sport = :$((CANDIDATE_BASE + OFFSET))" | grep -q .; then
                    BLOCK_FREE=0
                    break
                fi
            done
        fi
        if ((BLOCK_FREE)); then
            PORT_LOCK="$CANDIDATE_LOCK"
            PORT_BASE="$CANDIDATE_BASE"
            break
        fi
        rmdir "$CANDIDATE_LOCK" 2>/dev/null || true
    fi
done
if [[ -z "$PORT_LOCK" ]]; then echo 'ERROR: no free ML-Agents port block.' >&2; exit 1; fi
cleanup_port_lock() { rmdir "$PORT_LOCK" 2>/dev/null || true; }
trap cleanup_port_lock EXIT
trap 'cleanup_port_lock; exit 143' INT TERM

echo "Training $RUN_ID: $METHOD seed=$SEED cars=$CARS envs=$ENVS target-agent-steps=$STEPS resume=$RESUME"
echo "Output: $RUN_DIR"
echo "Port block: $PORT_BASE-$((PORT_BASE + PORT_BLOCK - 1))"
cat "$BUILD_INFO"
LEARN_EXTRA=()
if [[ "$RESUME" == 1 ]]; then LEARN_EXTRA+=(--resume); fi
apptainer exec \
    --bind "$STATS_PATCH:/usr/local/lib/python3.10/dist-packages/mlagents/trainers/stats.py" \
    --bind "$TRAINER_PATCH:/usr/local/lib/python3.10/dist-packages/mlagents/trainers/trainer_controller.py" \
    "$SIF" xvfb-run -a mlagents-learn "$CONFIG" \
        --env="$BUILD" --run-id="$RUN_ID" --results-dir="$RESULTS" \
        --no-graphics --torch-device=cpu --num-envs="$ENVS" \
        --base-port="$PORT_BASE" --seed="$SEED" "${LEARN_EXTRA[@]}" \
        --env-args -job-worker-count 1 -traffic-car-count "$CARS"
test -s "$RUN_DIR/CarAgent.onnx"
if ((STEPS >= 1000000)); then
    python3 "$SELECTOR" --run-dir "$RUN_DIR" --max-steps "$STEPS" --validate-all
fi
echo "Done: $RUN_DIR/CarAgent.onnx"
