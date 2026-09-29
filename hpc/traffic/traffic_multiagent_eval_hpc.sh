#!/bin/bash
# Held-out checkpoint evaluation for each training density (2, 5, 10 cars).
# 900 trained checkpoints + 5 original baselines, one checkpoint per array task.
# Final checkpoints also get ten seeded-route-map trajectories: 1 det + 5 stoch.
# Submit three copies of this array with TRAFFIC_EVAL_CARS=2, 5, and 10.

#SBATCH --job-name=traffic_ckpt
#SBATCH --partition=all
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --array=0-904%2
#SBATCH --output=/d/hpc/home/kk42117/mag/car1/logs/traffic_compare_%A_%a.out
#SBATCH --error=/d/hpc/home/kk42117/mag/car1/logs/traffic_compare_%A_%a.err

set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1

ROOT=/d/hpc/home/kk42117/mag/car1
SIF="$ROOT/mlagents.sif"
BUILD="$ROOT/Linux_traffic_all_validation/car.x86_64"
BUILD_INFO="$ROOT/Linux_traffic_all_validation/build_info.txt"
MANIFEST="$ROOT/traffic_validation_maps.json"
EVALUATOR="$ROOT/eval_traffic.py"
SELECTOR="$ROOT/select_multiagent_checkpoint.py"
PLOTTER="$ROOT/plot_multiagent_trajectories.py"
MAPS_DIR=${TRAFFIC_MAPS_DIR:-$ROOT/VoronoiVal}
TASK=${SLURM_ARRAY_TASK_ID:?Submit as a Slurm array}
TRAIN_ARRAY_ID=${TRAIN_ARRAY_ID:?Export TRAIN_ARRAY_ID from the training submission}
EVAL_CARS=${TRAFFIC_EVAL_CARS:?Set TRAFFIC_EVAL_CARS=2, 5 or 10}
EPISODES=${TRAFFIC_EVAL_EPISODES:-250}
if ((TASK < 0 || TASK > 904)) || ! [[ "$TRAIN_ARRAY_ID" =~ ^[0-9]+$ ]] || \
   ! [[ "$EVAL_CARS" =~ ^(2|5|10)$ ]] || ! [[ "$EPISODES" =~ ^[1-9][0-9]*$ ]]; then
    echo 'ERROR: invalid task, TRAIN_ARRAY_ID, TRAFFIC_EVAL_CARS or episode count.' >&2
    exit 1
fi
if ((TASK >= 900)); then
    METHOD=original
    TRAIN_CARS=0
    SEED=$((TASK - 900))
    MILESTONE=final
    RUN_ID="car_vgrid_19016936_laser12_cap3_s${SEED}"
    MODEL="$ROOT/results/$RUN_ID/CarAgent.onnx"
else
    DENSITIES=(2 5 10)
    TRAIN_CARS=${DENSITIES[$((TASK / 300))]}
    WITHIN=$((TASK % 300))
    if ((WITHIN < 100)); then
        METHOD=finetune
        SEED=$((WITHIN / 20))
        MILESTONE=$((WITHIN % 20 + 1))
        MAX_STEPS=20000000
    else
        METHOD=scratch
        SCRATCH_INDEX=$((WITHIN - 100))
        SEED=$((SCRATCH_INDEX / 40))
        MILESTONE=$((SCRATCH_INDEX % 40 + 1))
        MAX_STEPS=40000000
    fi
    RUN_ID="traffic${TRAIN_CARS}_${TRAIN_ARRAY_ID}_${METHOD}_s${SEED}"
    MODEL=$(python3 "$SELECTOR" --run-dir "$ROOT/results/$RUN_ID" \
        --max-steps "$MAX_STEPS" --milestone "$MILESTONE")
fi
RUN_DIR="$ROOT/results/$RUN_ID"
CONFIG="$RUN_DIR/configuration.yaml"
for required in "$SIF" "$BUILD" "$BUILD_INFO" "$MANIFEST" "$EVALUATOR" \
                "$SELECTOR" "$PLOTTER" "$MODEL" "$CONFIG"; do
    if [ ! -s "$required" ]; then echo "ERROR: missing or empty $required" >&2; exit 1; fi
done
if [[ "$MILESTONE" == final || ( "$METHOD" == finetune && "$MILESTONE" == 20 ) || \
      ( "$METHOD" == scratch && "$MILESTONE" == 40 ) ]]; then
    for map in curved_401 curved_406 curved_407 curved_438 curved_442 \
               curved_444 curved_451 curved_480 curved_484 curved_494; do
        if [[ ! -s "$MAPS_DIR/$map.txt" ]]; then
            echo "ERROR: trajectory map missing: $MAPS_DIR/$map.txt" >&2; exit 1
        fi
    done
fi
for line in 'Traffic measurement schema: multi-car-random-placement-v4' \
            'Traffic trajectory schema: per-seat-xz-step-v1' \
            'Traffic placement mode: random-connected-asphalt' \
            'Configured Vector Observation Size: 77  [MATCH]'; do
    if ! grep -Fq "$line" "$BUILD_INFO"; then
        echo "ERROR: validation build lacks '$line'." >&2; exit 1
    fi
done
if ! grep -Fq 'Traffic car pool size: 10;' "$BUILD_INFO"; then
    echo 'ERROR: validation build must support ten cars.' >&2; exit 1
fi

# Local port lock also excludes training tasks on the same node.
PORT_BLOCK=64
PORT_FIRST=10000
PORT_SLOT_COUNT=800
PORT_LOCK=
PREFERRED_SLOT=$(((SLURM_ARRAY_JOB_ID * 31 + TASK) % PORT_SLOT_COUNT))
for ((ATTEMPT=0; ATTEMPT<PORT_SLOT_COUNT; ATTEMPT++)); do
    SLOT=$(((PREFERRED_SLOT + ATTEMPT) % PORT_SLOT_COUNT))
    CANDIDATE_LOCK="/tmp/car_mlagents_portblock_${SLOT}.lock"
    if mkdir "$CANDIDATE_LOCK" 2>/dev/null; then
        PORT_LOCK="$CANDIDATE_LOCK"
        PORT_BASE=$((PORT_FIRST + SLOT * PORT_BLOCK))
        break
    fi
done
if [[ -z "$PORT_LOCK" ]]; then echo 'ERROR: no free ML-Agents port block.' >&2; exit 1; fi
cleanup_port_lock() { rmdir "$PORT_LOCK" 2>/dev/null || true; }
trap cleanup_port_lock EXIT
trap 'cleanup_port_lock; exit 143' INT TERM

OUT_ROOT="$ROOT/traffic_results/multiagent_checkpoints_${TRAIN_ARRAY_ID}/train${TRAIN_CARS}/${METHOD}_s${SEED}/step_${MILESTONE}/cars${EVAL_CARS}"
echo "Evaluating $RUN_ID checkpoint $MILESTONE ($MODEL), $EVAL_CARS cars, both modes."
echo "Output: $OUT_ROOT"
for MODE in deterministic stochastic; do
    OUT_DIR="$OUT_ROOT/${MODE}"
    mkdir -p "$OUT_DIR"
    cp "$BUILD_INFO" "$OUT_DIR/build_info.txt"
    sha256sum "$BUILD" "$MANIFEST" "$MODEL" "$CONFIG" > "$OUT_DIR/input_hashes.sha256"
    ARGS=(
        --build "$BUILD" --manifest "$MANIFEST" --models "$MODEL"
        --config "$CONFIG" --num-cars "$EVAL_CARS" --mode "$MODE"
        --episodes "$EPISODES" --require-tile-metrics --require-map-metrics
        --require-random-placement --base-port "$PORT_BASE" --worker-id-base 0
        --unity-job-worker-count 1 --time-scale 20 --no-graphics
        --out "$OUT_DIR/traffic.csv"
    )
    apptainer exec "$SIF" xvfb-run -a env \
        PYTHONPATH="/d/hpc/home/kk42117/.local/lib/python3.10/site-packages:${PYTHONPATH:-}" \
        python3 "$EVALUATOR" "${ARGS[@]}"
    echo "Done: $OUT_DIR/traffic.csv"

    # Match the single-car trajectory evaluation: the same ten held-out maps,
    # one deterministic and five stochastic rollouts. Placement seeds are fixed
    # across policies, modes, rollouts, and the 2/5/10-car route prefixes.
    if [[ "$MILESTONE" == final || ( "$METHOD" == finetune && "$MILESTONE" == 20 ) || \
          ( "$METHOD" == scratch && "$MILESTONE" == 40 ) ]]; then
        TRAJ_DIR="$OUT_DIR/trajectories"
        mkdir -p "$TRAJ_DIR"
        ROLLOUTS=1
        if [[ "$MODE" == stochastic ]]; then ROLLOUTS=5; fi
        TRAJ_ARGS=(
            --build "$BUILD" --manifest "$MANIFEST" --models "$MODEL"
            --config "$CONFIG" --num-cars "$EVAL_CARS" --mode "$MODE"
            --scenario-indices 0 5 6 37 41 43 50 79 83 93
            --rollouts-per-map "$ROLLOUTS" --fixed-placement-across-rollouts
            --trajectories-out "$TRAJ_DIR/trajectories.csv"
            --trajectory-stride-steps 5
            --require-tile-metrics --require-map-metrics --require-random-placement
            --base-port "$PORT_BASE" --worker-id-base 0
            --unity-job-worker-count 1 --time-scale 20 --no-graphics
            --out "$TRAJ_DIR/traffic.csv"
        )
        apptainer exec "$SIF" xvfb-run -a env \
            PYTHONPATH="/d/hpc/home/kk42117/.local/lib/python3.10/site-packages:${PYTHONPATH:-}" \
            python3 "$EVALUATOR" "${TRAJ_ARGS[@]}"
        echo "Done: $TRAJ_DIR/trajectories.csv"
    fi
done
if [[ "$MILESTONE" == final || ( "$METHOD" == finetune && "$MILESTONE" == 20 ) || \
      ( "$METHOD" == scratch && "$MILESTONE" == 40 ) ]]; then
    apptainer exec "$SIF" python3 "$PLOTTER" "$OUT_ROOT" \
        --maps-dir "$MAPS_DIR" --out "$OUT_ROOT/trajectory_grid.png"
fi
