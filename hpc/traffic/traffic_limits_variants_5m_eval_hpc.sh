#!/bin/bash
# 5M checkpoint evaluation for one traffic-limits variant.
#SBATCH --job-name=traffic_limits5m
#SBATCH --partition=all
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --array=0-4%5
#SBATCH --output=logs/traffic_limits5m_%A_%a.out
#SBATCH --error=logs/traffic_limits5m_%A_%a.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1

ROOT=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
SIF="$ROOT/mlagents.sif"; EVALUATOR="$ROOT/eval_traffic.py"; SELECTOR="$ROOT/select_multiagent_checkpoint.py"
MANIFEST="$ROOT/traffic_validation_maps.json"
VARIANT=${TRAFFIC_LIMITS_VARIANT:?submit wrapper must provide TRAFFIC_LIMITS_VARIANT}
TRAIN_JOB=${TRAFFIC_LIMITS_TRAIN_JOB_ID:?submit wrapper must provide TRAFFIC_LIMITS_TRAIN_JOB_ID}
SEED=${SLURM_ARRAY_TASK_ID:?array task required}; EPISODES=${TRAFFIC_CHECKPOINT_EVAL_EPISODES:-250}
STEPS=${TRAFFIC_LIMITS_STEPS:-40000000}
(( STEPS > 0 && STEPS % 5000000 == 0 )) || { echo 'ERROR: steps must be a positive multiple of 5M' >&2; exit 2; }
# Optional comma/space-separated list supports backfilling only missing
# checkpoints, e.g. TRAFFIC_EVAL_MILESTONES=45,50,55,60.  The default remains
# the complete 5M grid through the requested training budget.
if [[ -n "${TRAFFIC_EVAL_MILESTONES:-}" ]]; then
    milestone_text=${TRAFFIC_EVAL_MILESTONES//,/ }
    read -r -a MILESTONES <<< "$milestone_text"
    for m in "${MILESTONES[@]}"; do
        [[ "$m" =~ ^[0-9]+$ ]] && (( m > 0 && m <= STEPS / 1000000 && m % 5 == 0 )) || {
            echo "ERROR: invalid requested milestone '$m' for ${STEPS} steps" >&2; exit 2;
        }
    done
else
    MILESTONES=()
    for ((m=5; m<=STEPS/1000000; m+=5)); do MILESTONES+=("$m"); done
fi

case "$VARIANT" in
  time90_limits_128)
    RUN_PREFIX=traffic90_time_remove_finished_limits; OBS=90
    BUILD="$ROOT/Linux_traffic_all_validation_vehicle_signed_arcade_time/car.x86_64"
    INFO="$ROOT/Linux_traffic_all_validation_vehicle_signed_arcade_time/build_info.txt" ;;
  dense162_limits_128)
    RUN_PREFIX=traffic162_dense_time_remove_finished_limits128; OBS=162
    BUILD="$ROOT/Linux_traffic_all_validation_vehicle_signed_arcade_time_dense/car.x86_64"
    INFO="$ROOT/Linux_traffic_all_validation_vehicle_signed_arcade_time_dense/build_info.txt" ;;
  dense162_limits_256)
    RUN_PREFIX=traffic162_dense_time_remove_finished_limits256; OBS=162
    BUILD="$ROOT/Linux_traffic_all_validation_vehicle_signed_arcade_time_dense/car.x86_64"
    INFO="$ROOT/Linux_traffic_all_validation_vehicle_signed_arcade_time_dense/build_info.txt" ;;
  dense162_limits_256_stable)
    RUN_PREFIX=traffic162_dense_time_remove_finished_limits256_stable; OBS=162
    BUILD="$ROOT/Linux_traffic_all_validation_vehicle_signed_arcade_time_dense/car.x86_64"
    INFO="$ROOT/Linux_traffic_all_validation_vehicle_signed_arcade_time_dense/build_info.txt" ;;
  dense162_limits_256_middle)
    RUN_PREFIX=traffic162_dense_time_remove_finished_limits256_middle; OBS=162
    BUILD="$ROOT/Linux_traffic_all_validation_vehicle_signed_arcade_time_dense/car.x86_64"
    INFO="$ROOT/Linux_traffic_all_validation_vehicle_signed_arcade_time_dense/build_info.txt" ;;
  time90_limits_lagrangian)
    RUN_PREFIX=traffic90_time_remove_finished_limits_lagrangian; OBS=90
    BUILD="$ROOT/Linux_traffic_all_validation_vehicle_signed_arcade_time/car.x86_64"
    INFO="$ROOT/Linux_traffic_all_validation_vehicle_signed_arcade_time/build_info.txt" ;;
  time90_limits_constrained_ppo)
    RUN_PREFIX=traffic90_time_remove_finished_limits_cpo; OBS=90
    BUILD="$ROOT/Linux_traffic_all_validation_vehicle_signed_arcade_time/car.x86_64"
    INFO="$ROOT/Linux_traffic_all_validation_vehicle_signed_arcade_time/build_info.txt" ;;
  time90_limits_cpo_t12|time90_limits_cpo_t10|time90_limits_cpo_t08|time90_limits_cpo_t05)
    RUN_PREFIX="traffic90_time_remove_finished_limits_cpo_${VARIANT##*_t}"; OBS=90
    BUILD="$ROOT/Linux_traffic_all_validation_vehicle_signed_arcade_time/car.x86_64"
    INFO="$ROOT/Linux_traffic_all_validation_vehicle_signed_arcade_time/build_info.txt" ;;
  time90_limits_60)
    RUN_PREFIX=traffic90_time_remove_finished_limits_60; OBS=90
    BUILD="$ROOT/Linux_traffic_all_validation_vehicle_signed_arcade_time/car.x86_64"
    INFO="$ROOT/Linux_traffic_all_validation_vehicle_signed_arcade_time/build_info.txt" ;;
  time90_limits_cpo_t10_60|time90_limits_cpo_t08_60|time90_limits_cpo_t10_ttc_40|time90_limits_cpo_t10_ttc_60|time90_limits_cpo_t08_ttc_40|time90_limits_cpo_t08_ttc_60)
    RUN_PREFIX="traffic90_time_remove_finished_limits_${VARIANT#time90_limits_}"; OBS=90
    BUILD="$ROOT/Linux_traffic_all_validation_vehicle_signed_arcade_time/car.x86_64"
    INFO="$ROOT/Linux_traffic_all_validation_vehicle_signed_arcade_time/build_info.txt" ;;
  time90_limits_top5_40)
    RUN_PREFIX=traffic90_time_remove_finished_limits_top5_40; OBS=90
    BUILD="$ROOT/Linux_traffic_all_validation_vehicle_signed_arcade_time/car.x86_64"
    INFO="$ROOT/Linux_traffic_all_validation_vehicle_signed_arcade_time/build_info.txt" ;;
  *) echo "ERROR: unknown TRAFFIC_LIMITS_VARIANT=$VARIANT" >&2; exit 2 ;;
esac
RUN_ID="${RUN_PREFIX}_${TRAIN_JOB}_s${SEED}"
RUN_DIR="$ROOT/results/$RUN_ID"; CONFIG="$RUN_DIR/configuration.yaml"
OUT_ROOT="$ROOT/traffic_results/${VARIANT}_5m_${SLURM_ARRAY_JOB_ID}/${RUN_ID}"

for f in "$SIF" "$BUILD" "$INFO" "$EVALUATOR" "$SELECTOR" "$MANIFEST" "$CONFIG"; do
    [[ -s "$f" ]] || { echo "ERROR: missing or empty $f" >&2; exit 1; }
done
for line in 'Traffic car pool size: 10;' 'Traffic map source: VoronoiVal' \
            'Grid: 66x66, cell size 1.5m, mapSource Voronoi' \
            "Configured Vector Observation Size: ${OBS}  [MATCH]" \
            'includeRemainingStepsObservation: True' \
            'Actions: 2 continuous, discrete branches [none]' \
            'Traffic control mode: signed-torque arcade'; do
    grep -Fq "$line" "$INFO" || { echo "ERROR: incompatible validation build lacks '$line'" >&2; exit 1; }
done

reserve_port() {
    local serial=$1 attempt slot lock base offset free preferred
    preferred=$(((SLURM_ARRAY_JOB_ID * 97 + SEED * 31 + serial * 13) % 800))
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
    MODEL=$(python3 "$SELECTOR" --run-dir "$RUN_DIR" --max-steps "$STEPS" --milestone "$MILESTONE")
    PIDS=""; LOCKS=""; INDEX=0; FAIL=0
    for CARS in 2 5 10; do
        for MODE in deterministic stochastic; do
            OUT="$OUT_ROOT/step_${MILESTONE}/cars${CARS}/${MODE}/traffic.csv"
            if [[ -s "$OUT" ]] && (( $(wc -l < "$OUT") == EPISODES + 1 )); then INDEX=$((INDEX + 1)); continue; fi
            PORT_INFO=$(reserve_port $((MILESTONE * 10 + INDEX))) || { echo 'ERROR: no free ML-Agents port block' >&2; exit 1; }
            PORT_BASE=${PORT_INFO%%|*}; LOCK=${PORT_INFO#*|}; LOCKS="$LOCKS $LOCK"
            mkdir -p "$(dirname "$OUT")"
            apptainer exec "$SIF" xvfb-run -a env \
                PYTHONPATH="/d/hpc/home/kk42117/.local/lib/python3.10/site-packages:${PYTHONPATH:-}" \
                python3 "$EVALUATOR" --build "$BUILD" --manifest "$MANIFEST" --models "$MODEL" --config "$CONFIG" \
                    --num-cars "$CARS" --mode "$MODE" --episodes "$EPISODES" --expected-observations "$OBS" \
                    --require-tile-metrics --require-map-metrics --require-random-placement \
                    --base-port "$PORT_BASE" --worker-id-base 0 --unity-job-worker-count 1 \
                    --time-scale 20 --no-graphics --out "$OUT" &
            PIDS="$PIDS $!"; INDEX=$((INDEX + 1))
        done
    done
    for PID in $PIDS; do wait "$PID" || FAIL=1; done
    for LOCK in $LOCKS; do rmdir "$LOCK" 2>/dev/null || true; done
    ((FAIL == 0)) || { echo "ERROR: evaluation failed at ${MILESTONE}M" >&2; exit 1; }
done
echo "Done: $RUN_ID -> $OUT_ROOT"
