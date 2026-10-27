#!/bin/bash
# Held-out 5M checkpoint curves for the two refined-map, dense-lidar variants.
# One task evaluates one seed: eight checkpoints × 2/5/10 cars × two modes.
#SBATCH --job-name=traffic162_refined5m
#SBATCH --partition=all
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --array=0-9%5
#SBATCH --output=logs/traffic162_refined5m_%A_%a.out
#SBATCH --error=logs/traffic162_refined5m_%A_%a.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1

ROOT=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
SIF="$ROOT/mlagents.sif"
BUILD="$ROOT/Linux_traffic_all_validation_vehicle_signed_arcade_time_refined_dense/car.x86_64"
INFO="$ROOT/Linux_traffic_all_validation_vehicle_signed_arcade_time_refined_dense/build_info.txt"
EVALUATOR="$ROOT/eval_traffic.py"; SELECTOR="$ROOT/select_multiagent_checkpoint.py"
MANIFEST="$ROOT/traffic_validation_maps.json"
TRAIN_JOB=${TRAFFIC_TRAIN_JOB_ID:?submit wrapper must provide TRAFFIC_TRAIN_JOB_ID}
TASK=${SLURM_ARRAY_TASK_ID:?array task required}
VARIANT_INDEX=$((TASK / 5)); SEED=$((TASK % 5))
VARIANTS=(time_remove_goal time_remove_goal_gravel)
VARIANT=${VARIANTS[$VARIANT_INDEX]:-}
[[ -n "$VARIANT" ]] || { echo "ERROR: invalid task $TASK" >&2; exit 1; }
EPISODES=${TRAFFIC_CHECKPOINT_EVAL_EPISODES:-250}
RUN_ID="traffic162_refined_${VARIANT}_${TRAIN_JOB}_s${SEED}"
RUN_DIR="$ROOT/results/$RUN_ID"; CONFIG="$RUN_DIR/configuration.yaml"
OUT_ROOT="$ROOT/traffic_results/refined162_time_5m_${SLURM_ARRAY_JOB_ID}/${RUN_ID}"
MILESTONES=(5 10 15 20 25 30 35 40)

for f in "$SIF" "$BUILD" "$INFO" "$EVALUATOR" "$SELECTOR" "$MANIFEST" "$CONFIG"; do
    [[ -s "$f" ]] || { echo "ERROR: missing or empty $f" >&2; exit 1; }
done
for line in 'Traffic car pool size: 10;' 'Traffic map source: VoronoiVal' \
            'Grid: 330x330, cell size 0.3m, mapSource Voronoi' \
            'Configured Vector Observation Size: 162  [MATCH]' \
            'includeRemainingStepsObservation: True' \
            'Actions: 2 continuous, discrete branches [none]' \
            'Traffic control mode: signed-torque arcade'; do
    grep -Fq "$line" "$INFO" || { echo "ERROR: 162-observation refined validation build lacks '$line'" >&2; exit 1; }
done
[[ "$EPISODES" =~ ^[1-9][0-9]*$ ]] || { echo 'ERROR: TRAFFIC_CHECKPOINT_EVAL_EPISODES must be positive' >&2; exit 1; }

reserve_port() {
    local serial=$1 attempt slot lock base offset free preferred
    preferred=$(((SLURM_ARRAY_JOB_ID * 97 + TASK * 31 + serial * 13) % 800))
    for ((attempt=0; attempt<800; attempt++)); do
        slot=$(((preferred + attempt) % 800)); lock="/tmp/car_mlagents_portblock_${slot}.lock"
        if mkdir "$lock" 2>/dev/null; then
            base=$((10000 + slot * 64)); free=1
            if command -v ss >/dev/null 2>&1; then
                for ((offset=0; offset<64; offset++)); do
                    if ss -H -ltn "sport = :$((base + offset))" | grep -q .; then free=0; break; fi
                done
            fi
            if ((free)); then printf '%s|%s\n' "$base" "$lock"; return 0; fi
            rmdir "$lock" 2>/dev/null || true
        fi
    done
    return 1
}

for MILESTONE in "${MILESTONES[@]}"; do
    MODEL=$(python3 "$SELECTOR" --run-dir "$RUN_DIR" --max-steps 40000000 --milestone "$MILESTONE")
    PIDS=""; LOCKS=""; INDEX=0; FAIL=0
    for CARS in 2 5 10; do
        for MODE in deterministic stochastic; do
            OUT="$OUT_ROOT/step_${MILESTONE}/cars${CARS}/${MODE}/traffic.csv"
            if [[ -s "$OUT" ]] && (( $(wc -l < "$OUT") == EPISODES + 1 )); then INDEX=$((INDEX + 1)); continue; fi
            PORT_INFO=$(reserve_port $((MILESTONE * 10 + INDEX))) || { echo 'ERROR: no free evaluation port block' >&2; exit 1; }
            PORT_BASE=${PORT_INFO%%|*}; LOCK=${PORT_INFO#*|}; LOCKS="$LOCKS $LOCK"
            mkdir -p "$(dirname "$OUT")"
            echo "Checkpoint ${MILESTONE}M: $RUN_ID, $CARS cars, $MODE, port $PORT_BASE"
            apptainer exec "$SIF" xvfb-run -a env \
                PYTHONPATH="/d/hpc/home/kk42117/.local/lib/python3.10/site-packages:${PYTHONPATH:-}" \
                python3 "$EVALUATOR" --build "$BUILD" --manifest "$MANIFEST" \
                    --models "$MODEL" --config "$CONFIG" --num-cars "$CARS" --mode "$MODE" \
                    --episodes "$EPISODES" --expected-observations 162 \
                    --require-tile-metrics --require-map-metrics --require-random-placement \
                    --base-port "$PORT_BASE" --worker-id-base 0 --unity-job-worker-count 1 \
                    --time-scale 20 --no-graphics --out "$OUT" &
            PIDS="$PIDS $!"; INDEX=$((INDEX + 1))
        done
    done
    for PID in $PIDS; do wait "$PID" || FAIL=1; done
    for LOCK in $LOCKS; do rmdir "$LOCK" 2>/dev/null || true; done
    ((FAIL == 0)) || { echo "ERROR: evaluation failed at ${MILESTONE}M for $RUN_ID" >&2; exit 1; }
    echo "Completed $RUN_ID checkpoint ${MILESTONE}M"
done
echo "Done: 5M checkpoint evaluation for $RUN_ID -> $OUT_ROOT"
