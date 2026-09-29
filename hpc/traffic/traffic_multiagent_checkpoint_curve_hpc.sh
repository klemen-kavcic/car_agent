#!/bin/bash
# Post-training 77-observation learning curves. One task owns one trained run
# and evaluates every 1M checkpoint at 2/5/10 test cars in deterministic and
# stochastic mode. Final trajectories are already written by the training job.
# Submit only after all 30 training tasks complete:
# sbatch --export=ALL,TRAIN_JOB_ID=<traffic77_full_job_id> traffic_multiagent_checkpoint_curve_hpc.sh
#SBATCH --job-name=traffic77_curve
#SBATCH --partition=all
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=72:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --array=0-29%3
#SBATCH --output=/d/hpc/home/kk42117/mag/car1/logs/traffic77_curve_%A_%a.out
#SBATCH --error=/d/hpc/home/kk42117/mag/car1/logs/traffic77_curve_%A_%a.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1

ROOT=/d/hpc/home/kk42117/mag/car1
SIF="$ROOT/mlagents.sif"
BUILD="$ROOT/Linux_traffic_all_validation/car.x86_64"
INFO="$ROOT/Linux_traffic_all_validation/build_info.txt"
MANIFEST="$ROOT/traffic_validation_maps.json"
EVALUATOR="$ROOT/eval_traffic.py"
SELECTOR="$ROOT/select_multiagent_checkpoint.py"
TRAIN_JOB_ID=${TRAIN_JOB_ID:?Export the completed traffic77_full Slurm job ID as TRAIN_JOB_ID}
TASK=${SLURM_ARRAY_TASK_ID:?array task required}
EPISODES=${TRAFFIC_EVAL_EPISODES:-250}
((TASK >= 0 && TASK < 30)) || { echo 'ERROR: task must be 0..29' >&2; exit 1; }
[[ "$TRAIN_JOB_ID" =~ ^[0-9]+$ && "$EPISODES" =~ ^[1-9][0-9]*$ ]] || { echo 'ERROR: invalid TRAIN_JOB_ID or TRAFFIC_EVAL_EPISODES' >&2; exit 1; }

DENSITIES=(2 5 10); GROUP=$((TASK / 10)); WITHIN=$((TASK % 10)); TRAIN_CARS=${DENSITIES[$GROUP]}
if ((WITHIN < 5)); then METHOD=finetune; SEED=$WITHIN; MAX_STEPS=20000000; else METHOD=scratch; SEED=$((WITHIN-5)); MAX_STEPS=40000000; fi
RUN_ID="traffic${TRAIN_CARS}_${TRAIN_JOB_ID}_${METHOD}_s${SEED}"
RUN_DIR="$ROOT/results/$RUN_ID"
CONFIG="$RUN_DIR/configuration.yaml"
OUT_ROOT="$ROOT/traffic_results/multiagent_checkpoint_curves_${TRAIN_JOB_ID}/${RUN_ID}"

for f in "$SIF" "$BUILD" "$INFO" "$MANIFEST" "$EVALUATOR" "$SELECTOR" "$CONFIG" "$RUN_DIR/CarAgent.onnx"; do
    [[ -s "$f" ]] || { echo "ERROR: missing or empty $f" >&2; exit 1; }
done
for line in 'Traffic measurement schema: multi-car-random-placement-v4' 'Traffic car pool size: 10;' 'Configured Vector Observation Size: 77  [MATCH]'; do
    grep -Fq "$line" "$INFO" || { echo "ERROR: validation build lacks '$line'" >&2; exit 1; }
done

# Pick an unused 64-port block before any evaluator starts.
PORT_BLOCK=64; PORT_FIRST=10000; PORT_SLOT_COUNT=800; PORT_LOCK=
PREFERRED_SLOT=$(((SLURM_ARRAY_JOB_ID * 31 + TASK) % PORT_SLOT_COUNT))
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

for ((MILESTONE=1; MILESTONE<=MAX_STEPS/1000000; MILESTONE++)); do
    MODEL=$(python3 "$SELECTOR" --run-dir "$RUN_DIR" --max-steps "$MAX_STEPS" --milestone "$MILESTONE")
    [[ -s "$MODEL" ]] || { echo "ERROR: missing selected model for step $MILESTONE" >&2; exit 1; }
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
echo "Checkpoint curve complete: $OUT_ROOT"
