#!/bin/bash
# Evaluate the five historical single-agent laser12, 3 m/s-cap runs at each
# 1M checkpoint in traffic. This is the baseline learning curve for comparison
# with the new multi-agent-from-scratch curves.
# Submit: sbatch traffic_original_laser12_cap3_curve_hpc.sh
#SBATCH --job-name=traffic_old3_curve
#SBATCH --partition=all
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=72:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --array=0-4%3
#SBATCH --output=/d/hpc/home/kk42117/mag/car1/logs/traffic_old3_curve_%A_%a.out
#SBATCH --error=/d/hpc/home/kk42117/mag/car1/logs/traffic_old3_curve_%A_%a.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1

ROOT=/d/hpc/home/kk42117/mag/car1
SIF="$ROOT/mlagents.sif"
BUILD="$ROOT/Linux_traffic_all_validation/car.x86_64"
INFO="$ROOT/Linux_traffic_all_validation/build_info.txt"
MANIFEST="$ROOT/traffic_validation_maps.json"
EVALUATOR="$ROOT/eval_traffic.py"
SELECTOR="$ROOT/select_multiagent_checkpoint.py"
SEED=${SLURM_ARRAY_TASK_ID:?array task required}
EPISODES=${TRAFFIC_EVAL_EPISODES:-250}
((SEED >= 0 && SEED <= 4)) || { echo 'ERROR: seed must be 0..4' >&2; exit 1; }
[[ "$EPISODES" =~ ^[1-9][0-9]*$ ]] || { echo 'ERROR: TRAFFIC_EVAL_EPISODES must be positive' >&2; exit 1; }

RUN_ID="car_vgrid_19016936_laser12_cap3_s${SEED}"
RUN_DIR="$ROOT/results/$RUN_ID"
CONFIG="$RUN_DIR/configuration.yaml"
OUT_ROOT="$ROOT/traffic_results/original_laser12_cap3_checkpoint_curves_${SLURM_ARRAY_JOB_ID}/${RUN_ID}"
for f in "$SIF" "$BUILD" "$INFO" "$MANIFEST" "$EVALUATOR" "$SELECTOR" "$CONFIG" "$RUN_DIR/CarAgent.onnx"; do
    [[ -s "$f" ]] || { echo "ERROR: missing or empty $f" >&2; exit 1; }
done
for line in 'Traffic measurement schema: multi-car-random-placement-v4' 'Traffic car pool size: 10;' 'Configured Vector Observation Size: 77  [MATCH]'; do
    grep -Fq "$line" "$INFO" || { echo "ERROR: validation build lacks '$line'" >&2; exit 1; }
done
python3 "$SELECTOR" --run-dir "$RUN_DIR" --max-steps 20000000 --validate-all

PORT_BLOCK=64; PORT_FIRST=10000; PORT_SLOT_COUNT=800; PORT_LOCK=
PREFERRED_SLOT=$(((SLURM_ARRAY_JOB_ID * 31 + SEED) % PORT_SLOT_COUNT))
for ((ATTEMPT=0; ATTEMPT<PORT_SLOT_COUNT; ATTEMPT++)); do
    SLOT=$(((PREFERRED_SLOT + ATTEMPT) % PORT_SLOT_COUNT)); LOCK="/tmp/car_mlagents_portblock_${SLOT}.lock"
    if mkdir "$LOCK" 2>/dev/null; then
        BASE=$((PORT_FIRST + SLOT * PORT_BLOCK)); FREE=1
        if command -v ss >/dev/null 2>&1; then
            for ((OFFSET=0; OFFSET<PORT_BLOCK; OFFSET++)); do
                if ss -H -ltn "sport = :$((BASE + OFFSET))" | grep -q .; then FREE=0; break; fi
            done
        fi
        if ((FREE)); then PORT_LOCK="$LOCK"; PORT_BASE="$BASE"; break; fi
        rmdir "$LOCK" 2>/dev/null || true
    fi
done
[[ -n "$PORT_LOCK" ]] || { echo 'ERROR: no free ML-Agents port block' >&2; exit 1; }
cleanup(){ rmdir "$PORT_LOCK" 2>/dev/null || true; }
trap cleanup EXIT; trap 'cleanup; exit 143' INT TERM

for ((MILESTONE=1; MILESTONE<=20; MILESTONE++)); do
    MODEL=$(python3 "$SELECTOR" --run-dir "$RUN_DIR" --max-steps 20000000 --milestone "$MILESTONE")
    for TEST_CARS in 2 5 10; do
        for MODE in deterministic stochastic; do
            OUT="$OUT_ROOT/step_${MILESTONE}/cars${TEST_CARS}/${MODE}"
            mkdir -p "$OUT"
            echo "Evaluating $RUN_ID: step ${MILESTONE}M, test cars=$TEST_CARS, mode=$MODE"
            apptainer exec "$SIF" xvfb-run -a env \
                PYTHONPATH="/d/hpc/home/kk42117/.local/lib/python3.10/site-packages:${PYTHONPATH:-}" \
                python3 "$EVALUATOR" \
                    --build "$BUILD" --manifest "$MANIFEST" --models "$MODEL" --config "$CONFIG" \
                    --num-cars "$TEST_CARS" --mode "$MODE" --episodes "$EPISODES" \
                    --expected-observations 77 --require-tile-metrics --require-map-metrics \
                    --require-random-placement --base-port "$PORT_BASE" --worker-id-base 0 \
                    --unity-job-worker-count 1 --time-scale 20 --no-graphics --out "$OUT/traffic.csv"
        done
    done
done
echo "Historical 3 m/s learning curve complete: $OUT_ROOT"
