#!/bin/bash
# Controlled longer-time-limit evaluation of the original 89-observation,
# 40M, 10-car multi-agent scratch policy.  One array task evaluates one of
# the five policy seeds.  Deterministic and stochastic 10-car conditions run
# concurrently within that task, using separate reserved ML-Agents ports.
#
# This intentionally evaluates ONLY the final 40M ONNX.  Evaluating every
# 1M checkpoint would answer a different question and multiply this targeted
# 5,000-vs-10,000-step test by forty.
#
# Requires a validation build made after TrafficScenarioManager's
# traffic_shared_max_steps override was added.
#
# Submit:
#   sbatch traffic89_scratch40m_maxstep10000_eval_hpc.sh
#
# Optional quicker smoke check:
#   sbatch --array=0 --export=ALL,TRAFFIC_EVAL_EPISODES=5 traffic89_scratch40m_maxstep10000_eval_hpc.sh
#SBATCH --job-name=traffic89_10k
#SBATCH --partition=all
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --array=0-4%5
#SBATCH --output=/d/hpc/home/kk42117/mag/car1/logs/traffic89_10k_%A_%a.out
#SBATCH --error=/d/hpc/home/kk42117/mag/car1/logs/traffic89_10k_%A_%a.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1

ROOT=/d/hpc/home/kk42117/mag/car1
SIF="$ROOT/mlagents.sif"
BUILD="$ROOT/Linux_traffic_all_validation_vehicle/car.x86_64"
INFO="$ROOT/Linux_traffic_all_validation_vehicle/build_info.txt"
EVALUATOR="$ROOT/eval_traffic.py"
MANIFEST="$ROOT/traffic_validation_maps.json"
SEED=${SLURM_ARRAY_TASK_ID:?array task required}
EPISODES=${TRAFFIC_EVAL_EPISODES:-250}
SHARED_MAX_STEPS=${TRAFFIC_SHARED_MAX_STEPS:-10000}

case "$SEED" in
    0|1|2|4) RUN_ID="traffic89_19416041_scratch_cars10_s${SEED}" ;;
    3)     RUN_ID="traffic89_19416041_rerun_scratch_cars10_s3" ;;
    *) echo "ERROR: seed must be 0..4" >&2; exit 1 ;;
esac
RUN_DIR="$ROOT/results/$RUN_ID"
MODEL="$RUN_DIR/CarAgent.onnx"
CONFIG="$RUN_DIR/configuration.yaml"
OUT_ROOT="$ROOT/traffic_results/maxstep_${SHARED_MAX_STEPS}_${SLURM_ARRAY_JOB_ID}/${RUN_ID}/cars10"

for f in "$SIF" "$BUILD" "$INFO" "$EVALUATOR" "$MANIFEST" "$MODEL" "$CONFIG"; do
    [[ -s "$f" ]] || { echo "ERROR: missing or empty $f" >&2; exit 1; }
done
grep -Fq 'Configured Vector Observation Size: 89  [MATCH]' "$INFO" || {
    echo 'ERROR: validation build is not the 89-observation build' >&2; exit 1;
}
[[ "$EPISODES" =~ ^[1-9][0-9]*$ ]] || { echo 'ERROR: TRAFFIC_EVAL_EPISODES must be positive' >&2; exit 1; }
[[ "$SHARED_MAX_STEPS" =~ ^[1-9][0-9]*$ ]] || { echo 'ERROR: TRAFFIC_SHARED_MAX_STEPS must be positive' >&2; exit 1; }

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

PIDS=""; LOCKS=""; INDEX=0
for MODE in deterministic stochastic; do
    OUT="$OUT_ROOT/$MODE/traffic.csv"
    if [[ -s "$OUT" ]] && (( $(wc -l < "$OUT") == EPISODES + 1 )); then
        echo "Already complete: $RUN_ID, 10 cars, $MODE"
        INDEX=$((INDEX + 1)); continue
    fi
    PORT_INFO=$(reserve_port "$INDEX") || { echo 'ERROR: no free ML-Agents port block' >&2; exit 1; }
    PORT_BASE=${PORT_INFO%%|*}; LOCK=${PORT_INFO#*|}; LOCKS="$LOCKS $LOCK"
    mkdir -p "$(dirname "$OUT")"
    echo "Evaluating $RUN_ID: 10 cars, $MODE, $EPISODES episodes, shared budget $SHARED_MAX_STEPS"
    apptainer exec "$SIF" xvfb-run -a env \
        PYTHONPATH="/d/hpc/home/kk42117/.local/lib/python3.10/site-packages:${PYTHONPATH:-}" \
        python3 "$EVALUATOR" --build "$BUILD" --manifest "$MANIFEST" \
        --models "$MODEL" --config "$CONFIG" --num-cars 10 --mode "$MODE" \
        --episodes "$EPISODES" --expected-observations 89 \
        --shared-max-steps "$SHARED_MAX_STEPS" --max-engine-steps $((SHARED_MAX_STEPS + 20)) \
        --require-tile-metrics --require-map-metrics --require-random-placement \
        --base-port "$PORT_BASE" --worker-id-base 0 --unity-job-worker-count 1 \
        --time-scale 20 --no-graphics --out "$OUT" &
    PIDS="$PIDS $!"; INDEX=$((INDEX + 1))
done

FAIL=0
for PID in $PIDS; do wait "$PID" || FAIL=1; done
for LOCK in $LOCKS; do rmdir "$LOCK" 2>/dev/null || true; done
((FAIL == 0)) || { echo 'ERROR: one or more evaluations failed' >&2; exit 1; }
echo "Completed 10,000-step evaluation: $OUT_ROOT"
