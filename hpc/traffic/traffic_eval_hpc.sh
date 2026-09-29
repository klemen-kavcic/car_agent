#!/bin/bash
# Two measured two-car passes: deterministic and stochastic actions from the
# 3 m/s laser77 seed-0 model in both seats, across the same ten fixed cases.
# Submit with sbatch; optional smoke test: sbatch --export=ALL,TRAFFIC_SCENARIO=0 ...

#SBATCH --job-name=traffic_laser77_eval
#SBATCH --partition=all
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --signal=B:TERM@60
#SBATCH --output=/d/hpc/home/kk42117/mag/car1/logs/traffic_eval_%j.out
#SBATCH --error=/d/hpc/home/kk42117/mag/car1/logs/traffic_eval_%j.err

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
MODEL="$ROOT/results/car_vgrid_19016936_laser12_cap3_s0/CarAgent.onnx"
CONFIG="$ROOT/results/car_vgrid_19016936_laser12_cap3_s0/configuration.yaml"
OUT_DIR="$ROOT/traffic_results/${SLURM_JOB_ID:?This script must run under sbatch}"

for required in "$SIF" "$BUILD" "$BUILD_INFO" "$EVAL_SCRIPT" "$EVAL_HELPER" "$MODEL" "$CONFIG"; do
    if [ ! -f "$required" ]; then
        echo "ERROR: required input is missing: $required" >&2
        exit 1
    fi
done
if ! grep -Fq 'Tile observation mode: ContinuousLasers, 12 directions x 5 values = 60' "$BUILD_INFO" || \
   ! grep -Fq 'Configured Vector Observation Size: 77  [MATCH]' "$BUILD_INFO"; then
    echo 'ERROR: traffic build does not report the expected 12-ray/77-observation policy.' >&2
    cat "$BUILD_INFO" >&2
    exit 1
fi
if [ -n "${TRAFFIC_SCENARIO:-}" ] && ! [[ "$TRAFFIC_SCENARIO" =~ ^[0-9]$ ]]; then
    echo 'ERROR: TRAFFIC_SCENARIO must be a single digit 0-9.' >&2
    exit 1
fi

# Atomically reserve the same per-node port blocks used by the existing car jobs.
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
sha256sum "$BUILD" "$MODEL" "$CONFIG" "$EVAL_SCRIPT" "$EVAL_HELPER" > "$OUT_DIR/input_hashes.sha256"

echo "Traffic evaluation job $SLURM_JOB_ID"
echo "Output directory: $OUT_DIR"
echo "Reserved local port block: $PORT_BASE-$((PORT_BASE + PORT_BLOCK - 1))"
echo "Model in both seats: $MODEL"
echo 'Action modes: deterministic, stochastic'
if [ -n "${TRAFFIC_SCENARIO:-}" ]; then
    echo "Smoke-test scenario: $TRAFFIC_SCENARIO"
else
    echo 'Scenarios: all ten (0-9)'
fi
cat "$BUILD_INFO"

ARGS=(
    --build "$BUILD"
    --model0 "$MODEL"
    --model1 "$MODEL"
    --config "$CONFIG"
    --base-port "$PORT_BASE"
    --worker-id-base 0
    --unity-job-worker-count 1
    --time-scale 20
    --no-graphics
)
if grep -Fq 'Traffic measurement schema: finish-tile-and-per-seat-tile-seconds-v2' "$BUILD_INFO"; then
    ARGS+=(--require-tile-metrics)
fi
if [ -n "${TRAFFIC_SCENARIO:-}" ]; then
    ARGS+=(--scenario "$TRAFFIC_SCENARIO")
fi

for MODE in deterministic stochastic; do
    OUTPUT="$OUT_DIR/traffic_seed0_seed0_${MODE}.csv"
    echo "Starting $MODE evaluation: $OUTPUT"
    apptainer exec "$SIF" xvfb-run -a env \
        PYTHONPATH="/d/hpc/home/kk42117/.local/lib/python3.10/site-packages:${PYTHONPATH:-}" \
        python3 "$EVAL_SCRIPT" "${ARGS[@]}" --mode "$MODE" --out "$OUTPUT"
done

echo "Done: both modes are in $OUT_DIR"
