#!/bin/bash
# Evaluate every saved 1M ONNX checkpoint after one of the new traffic89
# experiments completes. Each array task handles one seed and launches the six
# 250-episode conditions (2/5/10 cars x deterministic/stochastic) in parallel.
# Submit through submit_traffic89_new_experiments_pipeline.sh so the required
# TRAFFIC_TRAIN_JOB_ID and TRAFFIC_CURVE_FAMILY are supplied automatically.
#SBATCH --job-name=traffic89_curve
#SBATCH --partition=all
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --array=0-4
#SBATCH --output=/d/hpc/home/kk42117/mag/car1/logs/traffic89_curve_%A_%a.out
#SBATCH --error=/d/hpc/home/kk42117/mag/car1/logs/traffic89_curve_%A_%a.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1

ROOT=/d/hpc/home/kk42117/mag/car1
SIF="$ROOT/mlagents.sif"
BUILD="$ROOT/Linux_traffic_all_validation_vehicle/car.x86_64"
INFO="$ROOT/Linux_traffic_all_validation_vehicle/build_info.txt"
EVALUATOR="$ROOT/eval_traffic.py"
SELECTOR="$ROOT/select_multiagent_checkpoint.py"
MANIFEST="$ROOT/traffic_validation_maps.json"
TRAIN_JOB=${TRAFFIC_TRAIN_JOB_ID:?submit wrapper must provide TRAFFIC_TRAIN_JOB_ID}
FAMILY=${TRAFFIC_CURVE_FAMILY:?submit wrapper must provide TRAFFIC_CURVE_FAMILY}
SEED=${SLURM_ARRAY_TASK_ID:?array task required}
EPISODES=${TRAFFIC_CHECKPOINT_EVAL_EPISODES:-250}
MAX_M=40

case "$FAMILY" in
    goal-removal)
        RUN_ID="traffic89_remove_goal_${TRAIN_JOB}_cars10_s${SEED}"
        ;;
    2to5to10)
        RUN_ID="traffic89_2to5to10_${TRAIN_JOB}_s${SEED}"
        ;;
    smooth-count)
        RUN_ID="traffic89_smooth_count_${TRAIN_JOB}_s${SEED}"
        ;;
    smooth-route)
        RUN_ID="traffic89_smooth_route_${TRAIN_JOB}_s${SEED}"
        ;;
    recovery-unshaped|recovery-clearance)
        RECOVERY_VARIANT=${FAMILY#recovery-}
        RUN_ID="traffic89_recovery_${RECOVERY_VARIANT}_${TRAIN_JOB}_s${SEED}"
        ;;
    recovery-arcade)
        RUN_ID="traffic89_recovery_arcade_unshaped_${TRAIN_JOB}_s${SEED}"
        BUILD="$ROOT/Linux_traffic_all_validation_vehicle_arcade/car.x86_64"
        INFO="$ROOT/Linux_traffic_all_validation_vehicle_arcade/build_info.txt"
        ;;
    arcade-scratch)
        RUN_ID="traffic89_arcade_scratch_${TRAIN_JOB}_s${SEED}"
        BUILD="$ROOT/Linux_traffic_all_validation_vehicle_arcade/car.x86_64"
        INFO="$ROOT/Linux_traffic_all_validation_vehicle_arcade/build_info.txt"
        ;;
    *)
        echo "ERROR: unknown TRAFFIC_CURVE_FAMILY '$FAMILY'" >&2
        exit 1
        ;;
esac
RUN_DIR="$ROOT/results/$RUN_ID"
CONFIG="$RUN_DIR/configuration.yaml"
OUT_ROOT="$ROOT/traffic_results/checkpoint_curves_new/${RUN_ID}"

for f in "$SIF" "$BUILD" "$INFO" "$EVALUATOR" "$SELECTOR" "$MANIFEST" "$CONFIG"; do
    [[ -s "$f" ]] || { echo "ERROR: missing or empty $f" >&2; exit 1; }
done
grep -Fq 'Configured Vector Observation Size: 89  [MATCH]' "$INFO" || {
    echo 'ERROR: validation build is not the 89-observation build' >&2; exit 1;
}
[[ "$EPISODES" =~ ^[1-9][0-9]*$ ]] || { echo 'ERROR: TRAFFIC_CHECKPOINT_EVAL_EPISODES must be positive' >&2; exit 1; }
python3 "$SELECTOR" --run-dir "$RUN_DIR" --max-steps 40000000 --validate-all

reserve_port() {
    local serial=$1 attempt slot lock base offset free preferred
    preferred=$(((SLURM_ARRAY_JOB_ID * 97 + SEED * 31 + serial * 13) % 800))
    for ((attempt=0; attempt<800; attempt++)); do
        slot=$(((preferred + attempt) % 800))
        lock="/tmp/car_mlagents_portblock_${slot}.lock"
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

for ((MILESTONE=1; MILESTONE<=MAX_M; MILESTONE++)); do
    MODEL=$(python3 "$SELECTOR" --run-dir "$RUN_DIR" --max-steps 40000000 --milestone "$MILESTONE")
    PIDS=""; LOCKS=""; INDEX=0; FAIL=0
    for CARS in 2 5 10; do
        for MODE in deterministic stochastic; do
            OUT="$OUT_ROOT/step_${MILESTONE}/cars${CARS}/${MODE}/traffic.csv"
            if [[ -s "$OUT" ]] && (( $(wc -l < "$OUT") == EPISODES + 1 )); then
                INDEX=$((INDEX + 1)); continue
            fi
            PORT_INFO=$(reserve_port $((MILESTONE * 10 + INDEX))) || {
                echo 'ERROR: no free ML-Agents port block for evaluation' >&2; exit 1;
            }
            PORT_BASE=${PORT_INFO%%|*}; LOCK=${PORT_INFO#*|}
            LOCKS="$LOCKS $LOCK"
            mkdir -p "$(dirname "$OUT")"
            echo "Checkpoint ${MILESTONE}M: $RUN_ID, $CARS cars, $MODE"
            apptainer exec "$SIF" xvfb-run -a env \
                PYTHONPATH="/d/hpc/home/kk42117/.local/lib/python3.10/site-packages:${PYTHONPATH:-}" \
                python3 "$EVALUATOR" --build "$BUILD" --manifest "$MANIFEST" \
                --models "$MODEL" --config "$CONFIG" --num-cars "$CARS" --mode "$MODE" \
                --episodes "$EPISODES" --expected-observations 89 \
                --require-tile-metrics --require-map-metrics --require-random-placement \
                --base-port "$PORT_BASE" --worker-id-base 0 --unity-job-worker-count 1 \
                --time-scale 20 --no-graphics --out "$OUT" &
            PIDS="$PIDS $!"
            INDEX=$((INDEX + 1))
        done
    done
    for PID in $PIDS; do wait "$PID" || FAIL=1; done
    for LOCK in $LOCKS; do rmdir "$LOCK" 2>/dev/null || true; done
    ((FAIL == 0)) || { echo "ERROR: evaluation failed at ${MILESTONE}M" >&2; exit 1; }
    echo "Completed $RUN_ID checkpoint ${MILESTONE}M"
done
