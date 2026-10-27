#!/bin/bash
# Final-only evaluation of the 40M checkpoint. Six 250-episode conditions run
# concurrently per seed (2/5/10 cars x deterministic/stochastic).
#SBATCH --job-name=traffic89_reward_final
#SBATCH --partition=all
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --array=0-4%5
#SBATCH --output=/d/hpc/home/kk42117/mag/car1/logs/traffic89_reward_final_%A_%a.out
#SBATCH --error=/d/hpc/home/kk42117/mag/car1/logs/traffic89_reward_final_%A_%a.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1

ROOT=/d/hpc/home/kk42117/mag/car1
SIF="$ROOT/mlagents.sif"; BUILD="$ROOT/Linux_traffic_all_validation_vehicle_reward/car.x86_64"
INFO="$ROOT/Linux_traffic_all_validation_vehicle_reward/build_info.txt"; EVALUATOR="$ROOT/eval_traffic.py"
MANIFEST="$ROOT/traffic_validation_maps.json"; TRAIN_JOB=${TRAFFIC_TRAIN_JOB_ID:?training job ID required}
VARIANT=${TRAFFIC_REWARD_VARIANT:?idle or exponential required}; SEED=${SLURM_ARRAY_TASK_ID:?array task required}
EPISODES=${TRAFFIC_FINAL_EVAL_EPISODES:-250}; RUN_ID="traffic89_reward_${VARIANT}_${TRAIN_JOB}_s${SEED}"
RUN_DIR="$ROOT/results/$RUN_ID"; CONFIG="$RUN_DIR/configuration.yaml"
OUT_ROOT="$ROOT/traffic_results/reward_shaping_final_${SLURM_ARRAY_JOB_ID}/${RUN_ID}"

for f in "$SIF" "$BUILD" "$INFO" "$EVALUATOR" "$MANIFEST" "$RUN_DIR/CarAgent.onnx" "$CONFIG"; do [[ -s "$f" ]] || { echo "ERROR: missing $f" >&2; exit 1; }; done
for line in 'Traffic car pool size: 10;' 'Traffic map source: VoronoiVal' 'Configured Vector Observation Size: 89  [MATCH]' 'Actions: 2 continuous, discrete branches [2,2]' 'Traffic reward-shaping support:'; do
    grep -Fq "$line" "$INFO" || { echo "ERROR: validation build lacks '$line'" >&2; exit 1; }
done
[[ "$EPISODES" =~ ^[1-9][0-9]*$ ]] || { echo 'ERROR: TRAFFIC_FINAL_EVAL_EPISODES must be positive' >&2; exit 1; }

reserve_port(){ local serial=$1 attempt slot lock base offset free preferred; preferred=$(((SLURM_ARRAY_JOB_ID*97 + SEED*31 + serial*13)%800)); for ((attempt=0;attempt<800;attempt++)); do slot=$(((preferred+attempt)%800)); lock="/tmp/car_mlagents_portblock_${slot}.lock"; if mkdir "$lock" 2>/dev/null; then base=$((10000+slot*64)); free=1; if command -v ss >/dev/null 2>&1; then for ((offset=0;offset<64;offset++)); do if ss -H -ltn "sport = :$((base+offset))" | grep -q .; then free=0; break; fi; done; fi; if ((free)); then printf '%s|%s\n' "$base" "$lock"; return 0; fi; rmdir "$lock" 2>/dev/null || true; fi; done; return 1; }
PIDS=""; LOCKS=""; INDEX=0; FAIL=0
for CARS in 2 5 10; do for MODE in deterministic stochastic; do
    OUT="$OUT_ROOT/cars${CARS}/${MODE}/traffic.csv"
    if [[ -s "$OUT" ]] && (( $(wc -l < "$OUT") == EPISODES + 1 )); then INDEX=$((INDEX+1)); continue; fi
    PORT_INFO=$(reserve_port "$INDEX") || { echo 'ERROR: no free evaluation port block' >&2; exit 1; }; PORT_BASE=${PORT_INFO%%|*}; LOCK=${PORT_INFO#*|}; LOCKS="$LOCKS $LOCK"; mkdir -p "$(dirname "$OUT")"
    apptainer exec "$SIF" xvfb-run -a env PYTHONPATH="/d/hpc/home/kk42117/.local/lib/python3.10/site-packages:${PYTHONPATH:-}" python3 "$EVALUATOR" --build "$BUILD" --manifest "$MANIFEST" --models "$RUN_DIR/CarAgent.onnx" --config "$CONFIG" --num-cars "$CARS" --mode "$MODE" --episodes "$EPISODES" --expected-observations 89 --require-tile-metrics --require-map-metrics --require-random-placement --base-port "$PORT_BASE" --worker-id-base 0 --unity-job-worker-count 1 --time-scale 20 --no-graphics --out "$OUT" &
    PIDS="$PIDS $!"; INDEX=$((INDEX+1))
done; done
for PID in $PIDS; do wait "$PID" || FAIL=1; done; for LOCK in $LOCKS; do rmdir "$LOCK" 2>/dev/null || true; done
((FAIL == 0)) || { echo "ERROR: final evaluation failed for $RUN_ID" >&2; exit 1; }
echo "Done: final 40M evaluation for $RUN_ID"
