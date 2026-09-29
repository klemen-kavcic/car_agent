#!/bin/bash
# All 100 held-out VoronoiVal maps, 10 maps per shard, in both action modes.
# TRAFFIC_CARS=2..10 chooses the active count from the same ten-car Unity build.
# TRAFFIC_SEED_STRATEGY=same|mixed selects seed mixes per map from seeds 0..4.
# Default fixed strategy keeps TRAFFIC_MODEL_SEEDS=0 or explicit per-seat seeds.

#SBATCH --job-name=traffic_val_all
#SBATCH --partition=all
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --signal=B:TERM@60
#SBATCH --array=0-19
#SBATCH --output=/d/hpc/home/kk42117/mag/car1/logs/traffic_val_%A_%a.out
#SBATCH --error=/d/hpc/home/kk42117/mag/car1/logs/traffic_val_%A_%a.err

set -euo pipefail
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1

ROOT=/d/hpc/home/kk42117/mag/car1
SIF="$ROOT/mlagents.sif"
BUILD_DIR=${TRAFFIC_BUILD_DIR:-Linux_traffic_all_validation}
MANIFEST_FILE=${TRAFFIC_MANIFEST_FILE:-traffic_validation_maps.json}
BUILD="$ROOT/$BUILD_DIR/car.x86_64"
BUILD_INFO="$ROOT/$BUILD_DIR/build_info.txt"
MANIFEST="$ROOT/$MANIFEST_FILE"
EXPECTED_MANIFEST=${MANIFEST_FILE%.json}
EVAL_SCRIPT="$ROOT/eval_traffic.py"
EVAL_HELPER="$ROOT/eval_checkpoints.py"
CARS=${TRAFFIC_CARS:-2}
STRATEGY=${TRAFFIC_SEED_STRATEGY:-fixed}
ROLLOUTS=${TRAFFIC_ROLLOUTS_PER_MAP:-1}
PLACEMENT_SEED=${TRAFFIC_PLACEMENT_SEED:-3401}
ASSIGNMENT_SEED=${TRAFFIC_ASSIGNMENT_SEED:-7319}
TASK_ID=${SLURM_ARRAY_TASK_ID:?Submit this script with sbatch}

if ! [[ "$CARS" =~ ^[1-9][0-9]*$ ]] || ((CARS < 2)); then
    echo 'ERROR: TRAFFIC_CARS must be an integer >= 2.' >&2
    exit 1
fi
if ! [[ "$ROLLOUTS" =~ ^[1-9][0-9]*$ ]] || ! [[ "$PLACEMENT_SEED" =~ ^[0-9]+$ ]]; then
    echo 'ERROR: TRAFFIC_ROLLOUTS_PER_MAP must be positive and TRAFFIC_PLACEMENT_SEED nonnegative.' >&2
    exit 1
fi
if [[ "$STRATEGY" != fixed && "$STRATEGY" != same && "$STRATEGY" != mixed ]]; then
    echo 'ERROR: TRAFFIC_SEED_STRATEGY must be fixed, same or mixed.' >&2
    exit 1
fi
if [[ "$STRATEGY" != fixed && "$CARS" != 2 && "$CARS" != 5 && "$CARS" != 10 ]]; then
    echo 'ERROR: same/mixed seed comparison is defined for 2, 5 or 10 cars.' >&2
    exit 1
fi
if ! [[ "$ASSIGNMENT_SEED" =~ ^[0-9]+$ ]]; then
    echo 'ERROR: TRAFFIC_ASSIGNMENT_SEED must be nonnegative.' >&2
    exit 1
fi
if ((TASK_ID < 0 || TASK_ID > 19)); then
    echo "ERROR: array task $TASK_ID is outside 0-19." >&2
    exit 1
fi
for required in "$SIF" "$BUILD" "$BUILD_INFO" "$MANIFEST" "$EVAL_SCRIPT" "$EVAL_HELPER"; do
    if [ ! -f "$required" ]; then
        echo "ERROR: required input is missing: $required" >&2
        exit 1
    fi
done
if ! grep -Fq 'Traffic measurement schema: multi-car-random-placement-v4' "$BUILD_INFO" || \
   ! grep -Fq "Traffic scenario manifest: $EXPECTED_MANIFEST" "$BUILD_INFO" || \
   ! grep -Fq 'Traffic placement mode: random-connected-asphalt' "$BUILD_INFO" || \
   ! grep -Fq 'Configured Vector Observation Size: 77  [MATCH]' "$BUILD_INFO"; then
    echo 'ERROR: build info does not match the randomized full-validation traffic scene.' >&2
    cat "$BUILD_INFO" >&2
    exit 1
fi
POOL_SIZE=$(sed -n 's/^Traffic car pool size: \([0-9][0-9]*\);.*/\1/p' "$BUILD_INFO")
if [ -z "$POOL_SIZE" ] || ((CARS > POOL_SIZE)); then
    echo "ERROR: requested $CARS cars, but build pool is ${POOL_SIZE:-unknown}." >&2
    exit 1
fi

if [[ "$STRATEGY" == fixed ]]; then
    IFS=',' read -r -a SEEDS <<< "${TRAFFIC_MODEL_SEEDS:-0}"
    if ((${#SEEDS[@]} != 1 && ${#SEEDS[@]} != CARS)); then
        echo 'ERROR: TRAFFIC_MODEL_SEEDS needs one shared seed or one seed per car.' >&2
        exit 1
    fi
else
    SEEDS=(0 1 2 3 4)
fi
MODELS=()
for seed in "${SEEDS[@]}"; do
    if ! [[ "$seed" =~ ^[0-4]$ ]]; then
        echo "ERROR: unsupported seed '$seed'; expected 0..4." >&2
        exit 1
    fi
    run="$ROOT/results/car_vgrid_19016936_laser12_cap3_s${seed}"
    for required in "$run/CarAgent.onnx" "$run/configuration.yaml"; do
        if [ ! -s "$required" ]; then
            echo "ERROR: missing or empty model input: $required" >&2
            exit 1
        fi
    done
    MODELS+=("$run/CarAgent.onnx")
done

SHARD=$((TASK_ID / 2))
START=$((SHARD * 10))
STOP=$((START + 10))
if ((TASK_ID % 2 == 0)); then MODE=deterministic; else MODE=stochastic; fi
SEED_LABEL=$(IFS=-; echo "${SEEDS[*]}")
OUT_DIR="$ROOT/traffic_results/full_validation_${SLURM_ARRAY_JOB_ID}/cars${CARS}_${STRATEGY}_seeds${SEED_LABEL}/${MODE}/shard${SHARD}"
if [ -n "${TRAFFIC_SCENARIO:-}" ] && \
   ! [[ "$TRAFFIC_SCENARIO" =~ ^([0-9]|[1-9][0-9])$ ]]; then
    echo 'ERROR: TRAFFIC_SCENARIO must be an integer 0..99.' >&2
    exit 1
fi
EPISODES=${TRAFFIC_EPISODES:-}
if [[ "$STRATEGY" != fixed && -z "$EPISODES" ]]; then
    if [ -n "${TRAFFIC_SCENARIO:-}" ]; then EPISODES=1; else EPISODES=25; fi
fi
if [ -n "$EPISODES" ] && ! [[ "$EPISODES" =~ ^[1-9][0-9]*$ ]]; then
    echo 'ERROR: TRAFFIC_EPISODES must be a positive integer.' >&2
    exit 1
fi

# Same per-node, atomic 64-port reservation used by the earlier traffic sweep.
# eval_traffic.py retries a different port within this block on a bind collision.
PORT_BLOCK=64
PORT_FIRST=10000
PORT_SLOT_COUNT=800
PORT_LOCK=
PREFERRED_SLOT=$((SLURM_JOB_ID % PORT_SLOT_COUNT))
for ((ATTEMPT=0; ATTEMPT<PORT_SLOT_COUNT; ATTEMPT++)); do
    SLOT=$(((PREFERRED_SLOT + ATTEMPT) % PORT_SLOT_COUNT))
    CANDIDATE_LOCK="/tmp/car_mlagents_portblock_${SLOT}.lock"
    if mkdir "$CANDIDATE_LOCK" 2>/dev/null; then
        PORT_LOCK="$CANDIDATE_LOCK"
        PORT_BASE=$((PORT_FIRST + SLOT * PORT_BLOCK))
        break
    fi
done
if [ -z "$PORT_LOCK" ]; then
    echo 'ERROR: no free local ML-Agents port block.' >&2
    exit 1
fi
cleanup_port_lock() { rmdir "$PORT_LOCK" 2>/dev/null || true; }
trap cleanup_port_lock EXIT
trap 'cleanup_port_lock; exit 143' INT TERM

mkdir -p "$OUT_DIR"
cp "$BUILD_INFO" "$OUT_DIR/build_info.txt"
sha256sum "$BUILD" "$MANIFEST" "$EVAL_SCRIPT" "$EVAL_HELPER" "${MODELS[@]}" \
    > "$OUT_DIR/input_hashes.sha256"
echo "Array ${SLURM_ARRAY_JOB_ID} task $TASK_ID: cars=$CARS strategy=$STRATEGY seeds=$SEED_LABEL $MODE maps=$START-$((STOP - 1)) episodes=${EPISODES:-per-map}"
echo "Output: $OUT_DIR"
echo "Reserved local port block: $PORT_BASE-$((PORT_BASE + PORT_BLOCK - 1))"

ARGS=(
    --build "$BUILD"
    --manifest "$MANIFEST"
    --num-cars "$CARS"
    --mode "$MODE"
    --require-tile-metrics
    --require-map-metrics
    --require-random-placement
    --rollouts-per-map "$ROLLOUTS"
    --placement-seed "$PLACEMENT_SEED"
    --base-port "$PORT_BASE"
    --worker-id-base 0
    --unity-job-worker-count 1
    --time-scale 20
    --no-graphics
    --out "$OUT_DIR/traffic.csv"
)
if [[ "$STRATEGY" == fixed ]]; then
    ARGS+=(--models "${MODELS[@]}")
else
    ARGS+=(--seed-pool "${MODELS[@]}" --seed-strategy "$STRATEGY" \
           --assignment-seed "$ASSIGNMENT_SEED")
fi
if [ -n "$EPISODES" ]; then ARGS+=(--episodes "$EPISODES"); fi
if [ -n "${TRAFFIC_SCENARIO:-}" ]; then
    ARGS+=(--scenario "$TRAFFIC_SCENARIO")
else
    ARGS+=(--scenario-start "$START" --scenario-stop "$STOP")
fi

apptainer exec "$SIF" xvfb-run -a env \
    PYTHONPATH="/d/hpc/home/kk42117/.local/lib/python3.10/site-packages:${PYTHONPATH:-}" \
    python3 "$EVAL_SCRIPT" "${ARGS[@]}"

echo "Done: $OUT_DIR/traffic.csv"
