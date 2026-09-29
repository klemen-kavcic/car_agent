#!/bin/bash
# Run final held-out evaluation and ten-map trajectories within the training
# Slurm allocation. Called by the vehicle-distance training scripts; do not
# submit this file by itself.
set -euo pipefail

ROOT=${ROOT:?ROOT must be set by the training script}
SIF=${SIF:?SIF must be set by the training script}
RUN_ID=${1:?run ID required}
MODEL=${2:?ONNX model path required}
CONFIG=${3:?configuration path required}
JOB_ID=${SLURM_ARRAY_JOB_ID:?must run inside a Slurm array task}
TASK_ID=${SLURM_ARRAY_TASK_ID:?must run inside a Slurm array task}

BUILD="$ROOT/Linux_traffic_all_validation_vehicle/car.x86_64"
BUILD_INFO="$ROOT/Linux_traffic_all_validation_vehicle/build_info.txt"
MANIFEST="$ROOT/traffic_validation_maps.json"
EVALUATOR="$ROOT/eval_traffic.py"
PLOTTER="$ROOT/plot_multiagent_trajectories.py"
MAPS_DIR=${TRAFFIC_MAPS_DIR:-$ROOT/VoronoiVal}
EPISODES=${TRAFFIC_EVAL_EPISODES:-250}
OUT_ROOT="$ROOT/traffic_results/vehicle89_${JOB_ID}/${RUN_ID}"

for f in "$BUILD" "$BUILD_INFO" "$MANIFEST" "$EVALUATOR" "$PLOTTER" "$MODEL" "$CONFIG"; do
    [[ -s "$f" ]] || { echo "ERROR: missing or empty $f" >&2; exit 1; }
done
[[ "$EPISODES" =~ ^[1-9][0-9]*$ ]] || { echo 'ERROR: TRAFFIC_EVAL_EPISODES must be positive' >&2; exit 1; }
for line in \
    'Traffic measurement schema: multi-car-random-placement-v4' \
    'Traffic trajectory schema: per-seat-xz-step-v1' \
    'Traffic car pool size: 10;' \
    'Configured Vector Observation Size: 89  [MATCH]'; do
    grep -Fq "$line" "$BUILD_INFO" || { echo "ERROR: validation build lacks '$line'" >&2; exit 1; }
done
for map in curved_401 curved_406 curved_407 curved_438 curved_442 \
           curved_444 curved_451 curved_480 curved_484 curved_494; do
    [[ -s "$MAPS_DIR/$map.txt" ]] || { echo "ERROR: trajectory map missing: $MAPS_DIR/$map.txt" >&2; exit 1; }
done

# The parent training script holds its port lock until this helper returns, so
# the same local base port is safe after the trainer's Unity process exits.
PORT_BASE=${PORT_BASE:?PORT_BASE must be set by the training script}
for CARS in 2 5 10; do
    CONDITION_ROOT="$OUT_ROOT/cars${CARS}"
    for MODE in deterministic stochastic; do
        OUT_DIR="$CONDITION_ROOT/$MODE"
        mkdir -p "$OUT_DIR"
        if [[ -s "$OUT_DIR/traffic.csv" ]] && (( $(wc -l < "$OUT_DIR/traffic.csv") == EPISODES + 1 )); then
            echo "Final evaluation already complete: $RUN_ID, $CARS cars, $MODE"
        else
            echo "Final evaluation: $RUN_ID, $CARS cars, $MODE, $EPISODES episodes"
            apptainer exec "$SIF" xvfb-run -a env \
                PYTHONPATH="/d/hpc/home/kk42117/.local/lib/python3.10/site-packages:${PYTHONPATH:-}" \
                python3 "$EVALUATOR" \
                    --build "$BUILD" --manifest "$MANIFEST" --models "$MODEL" --config "$CONFIG" \
                    --num-cars "$CARS" --mode "$MODE" --episodes "$EPISODES" \
                    --expected-observations 89 --require-tile-metrics --require-map-metrics \
                    --require-random-placement --base-port "$PORT_BASE" --worker-id-base 0 \
                    --unity-job-worker-count 1 --time-scale 20 --no-graphics --out "$OUT_DIR/traffic.csv"
        fi

        TRAJ_DIR="$OUT_DIR/trajectories"
        mkdir -p "$TRAJ_DIR"
        ROLLOUTS=1
        [[ "$MODE" == stochastic ]] && ROLLOUTS=5
        apptainer exec "$SIF" xvfb-run -a env \
            PYTHONPATH="/d/hpc/home/kk42117/.local/lib/python3.10/site-packages:${PYTHONPATH:-}" \
            python3 "$EVALUATOR" \
                --build "$BUILD" --manifest "$MANIFEST" --models "$MODEL" --config "$CONFIG" \
                --num-cars "$CARS" --mode "$MODE" \
                --scenario-indices 0 5 6 37 41 43 50 79 83 93 \
                --rollouts-per-map "$ROLLOUTS" --fixed-placement-across-rollouts \
                --trajectories-out "$TRAJ_DIR/trajectories.csv" --trajectory-stride-steps 5 \
                --expected-observations 89 --require-tile-metrics --require-map-metrics \
                --require-random-placement --base-port "$PORT_BASE" --worker-id-base 0 \
                --unity-job-worker-count 1 --time-scale 20 --no-graphics --out "$TRAJ_DIR/traffic.csv"
    done
    apptainer exec "$SIF" python3 "$PLOTTER" "$CONDITION_ROOT" \
        --maps-dir "$MAPS_DIR" --out "$CONDITION_ROOT/trajectory_grid.png"
done
echo "Final 89-observation evaluation and trajectories: $OUT_ROOT"
