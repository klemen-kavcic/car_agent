#!/bin/bash
# Five cap-3 laser77 seeds in both ordered seats: 5 x 5 x 2 action modes.
# Each of the 50 array tasks evaluates all ten fixed traffic maps once.
# A/B and B/A are distinct because the two routes and spawns are distinct.

#SBATCH --job-name=traffic_seed_pairs
#SBATCH --partition=all
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --signal=B:TERM@60
#SBATCH --array=0-49%6
#SBATCH --output=/d/hpc/home/kk42117/mag/car1/logs/traffic_pairs_%A_%a.out
#SBATCH --error=/d/hpc/home/kk42117/mag/car1/logs/traffic_pairs_%A_%a.err

set -euo pipefail

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1

ROOT=/d/hpc/home/kk42117/mag/car1
SIF="$ROOT/mlagents.sif"
BUILD="$ROOT/Linux_traffic/car.x86_64"
BUILD_INFO="$ROOT/Linux_traffic/build_info.txt"
EVAL_SCRIPT="$ROOT/eval_traffic.py"
EVAL_HELPER="$ROOT/eval_checkpoints.py"
TASK_ID=${SLURM_ARRAY_TASK_ID:?Submit this script with sbatch}
if ((TASK_ID < 0 || TASK_ID > 49)); then
    echo "ERROR: array task $TASK_ID is outside 0-49" >&2
    exit 1
fi

PAIR=$((TASK_ID / 2))
SEAT0=$((PAIR / 5))
SEAT1=$((PAIR % 5))
if ((TASK_ID % 2 == 0)); then
    MODE=deterministic
else
    MODE=stochastic
fi

RUN0="car_vgrid_19016936_laser12_cap3_s${SEAT0}"
RUN1="car_vgrid_19016936_laser12_cap3_s${SEAT1}"
MODEL0="$ROOT/results/$RUN0/CarAgent.onnx"
MODEL1="$ROOT/results/$RUN1/CarAgent.onnx"
CONFIG0="$ROOT/results/$RUN0/configuration.yaml"
CONFIG1="$ROOT/results/$RUN1/configuration.yaml"
OUT_DIR="$ROOT/traffic_results/pairwise_${SLURM_ARRAY_JOB_ID}/s${SEAT0}_s${SEAT1}_${MODE}"

for required in "$SIF" "$BUILD" "$BUILD_INFO" "$EVAL_SCRIPT" "$EVAL_HELPER" \
                "$MODEL0" "$MODEL1" "$CONFIG0" "$CONFIG1"; do
    if [ ! -f "$required" ]; then
        echo "ERROR: required input is missing: $required" >&2
        exit 1
    fi
done
if ! grep -Fq 'Tile observation mode: ContinuousLasers, 12 directions x 5 values = 60' "$BUILD_INFO" || \
   ! grep -Fq 'Configured Vector Observation Size: 77  [MATCH]' "$BUILD_INFO" || \
   ! grep -Fq 'Traffic measurement schema: finish-tile-and-per-seat-tile-seconds-v2' "$BUILD_INFO"; then
    echo 'ERROR: rebuild TrafficTwoCar with the new per-seat tile metrics before this sweep.' >&2
    cat "$BUILD_INFO" >&2
    exit 1
fi
if [ -n "${TRAFFIC_SCENARIO:-}" ] && ! [[ "$TRAFFIC_SCENARIO" =~ ^[0-9]$ ]]; then
    echo 'ERROR: TRAFFIC_SCENARIO must be a single digit 0-9.' >&2
    exit 1
fi

# Same atomic per-node port reservation as the existing training/evaluation jobs.
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
sha256sum "$BUILD" "$MODEL0" "$MODEL1" "$CONFIG0" "$CONFIG1" \
          "$EVAL_SCRIPT" "$EVAL_HELPER" > "$OUT_DIR/input_hashes.sha256"

echo "Array job ${SLURM_ARRAY_JOB_ID}, task $TASK_ID: seat0 seed$SEAT0 vs seat1 seed$SEAT1, $MODE"
echo "Output directory: $OUT_DIR"
echo "Reserved local port block: $PORT_BASE-$((PORT_BASE + PORT_BLOCK - 1))"
cat "$BUILD_INFO"

ARGS=(
    --build "$BUILD"
    --model0 "$MODEL0"
    --model1 "$MODEL1"
    --config "$CONFIG0"
    --mode "$MODE"
    --require-tile-metrics
    --base-port "$PORT_BASE"
    --worker-id-base 0
    --unity-job-worker-count 1
    --time-scale 20
    --no-graphics
    --out "$OUT_DIR/traffic.csv"
)
if [ -n "${TRAFFIC_SCENARIO:-}" ]; then
    ARGS+=(--scenario "$TRAFFIC_SCENARIO")
fi

apptainer exec "$SIF" xvfb-run -a env \
    PYTHONPATH="/d/hpc/home/kk42117/.local/lib/python3.10/site-packages:${PYTHONPATH:-}" \
    python3 "$EVAL_SCRIPT" "${ARGS[@]}"

echo "Done: $OUT_DIR/traffic.csv"
