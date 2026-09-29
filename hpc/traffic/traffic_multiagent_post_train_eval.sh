#!/bin/bash
# Final 77-observation held-out evaluation and trajectory collection. This is
# invoked by traffic_multiagent_train_eval_hpc.sh within the same Slurm task.
set -euo pipefail

ROOT=${ROOT:?ROOT must be set}
SIF=${SIF:?SIF must be set}
RUN_ID=${1:?run ID required}
MODEL=${2:?ONNX model required}
CONFIG=${3:?configuration required}
JOB_ID=${SLURM_ARRAY_JOB_ID:?must run inside a Slurm array task}
PORT_BASE=${PORT_BASE:?PORT_BASE must be set}

BUILD="$ROOT/Linux_traffic_all_validation/car.x86_64"
INFO="$ROOT/Linux_traffic_all_validation/build_info.txt"
MANIFEST="$ROOT/traffic_validation_maps.json"
EVALUATOR="$ROOT/eval_traffic.py"
PLOTTER="$ROOT/plot_multiagent_trajectories.py"
MAPS_DIR=${TRAFFIC_MAPS_DIR:-$ROOT/VoronoiVal}
EPISODES=${TRAFFIC_EVAL_EPISODES:-250}
OUT_ROOT="$ROOT/traffic_results/multiagent_final_${JOB_ID}/${RUN_ID}"

for f in "$BUILD" "$INFO" "$MANIFEST" "$EVALUATOR" "$PLOTTER" "$MODEL" "$CONFIG"; do
    [[ -s "$f" ]] || { echo "ERROR: missing or empty $f" >&2; exit 1; }
done
[[ "$EPISODES" =~ ^[1-9][0-9]*$ ]] || { echo 'ERROR: TRAFFIC_EVAL_EPISODES must be positive' >&2; exit 1; }
for line in \
    'Traffic measurement schema: multi-car-random-placement-v4' \
    'Traffic trajectory schema: per-seat-xz-step-v1' \
    'Traffic car pool size: 10;' \
    'Configured Vector Observation Size: 77  [MATCH]'; do
    grep -Fq "$line" "$INFO" || { echo "ERROR: validation build lacks '$line'" >&2; exit 1; }
done
for map in curved_401 curved_406 curved_407 curved_438 curved_442 \
           curved_444 curved_451 curved_480 curved_484 curved_494; do
    [[ -s "$MAPS_DIR/$map.txt" ]] || { echo "ERROR: trajectory map missing: $MAPS_DIR/$map.txt" >&2; exit 1; }
done

for CARS in 2 5 10; do
    CONDITION="$OUT_ROOT/cars${CARS}"
    for MODE in deterministic stochastic; do
        OUT="$CONDITION/$MODE"
        mkdir -p "$OUT"
        echo "Final evaluation: $RUN_ID, $CARS cars, $MODE, $EPISODES episodes"
        apptainer exec "$SIF" xvfb-run -a env \
            PYTHONPATH="/d/hpc/home/kk42117/.local/lib/python3.10/site-packages:${PYTHONPATH:-}" \
            python3 "$EVALUATOR" \
                --build "$BUILD" --manifest "$MANIFEST" --models "$MODEL" --config "$CONFIG" \
                --num-cars "$CARS" --mode "$MODE" --episodes "$EPISODES" \
                --expected-observations 77 --require-tile-metrics --require-map-metrics \
                --require-random-placement --base-port "$PORT_BASE" --worker-id-base 0 \
                --unity-job-worker-count 1 --time-scale 20 --no-graphics --out "$OUT/traffic.csv"

        TRAJ="$OUT/trajectories"
        mkdir -p "$TRAJ"
        ROLLOUTS=1
        [[ "$MODE" == stochastic ]] && ROLLOUTS=5
        apptainer exec "$SIF" xvfb-run -a env \
            PYTHONPATH="/d/hpc/home/kk42117/.local/lib/python3.10/site-packages:${PYTHONPATH:-}" \
            python3 "$EVALUATOR" \
                --build "$BUILD" --manifest "$MANIFEST" --models "$MODEL" --config "$CONFIG" \
                --num-cars "$CARS" --mode "$MODE" \
                --scenario-indices 0 5 6 37 41 43 50 79 83 93 \
                --rollouts-per-map "$ROLLOUTS" --fixed-placement-across-rollouts \
                --trajectories-out "$TRAJ/trajectories.csv" --trajectory-stride-steps 5 \
                --expected-observations 77 --require-tile-metrics --require-map-metrics \
                --require-random-placement --base-port "$PORT_BASE" --worker-id-base 0 \
                --unity-job-worker-count 1 --time-scale 20 --no-graphics --out "$TRAJ/traffic.csv"
    done
    apptainer exec "$SIF" python3 "$PLOTTER" "$CONDITION" \
        --maps-dir "$MAPS_DIR" --out "$CONDITION/trajectory_grid.png"
done

echo "Final evaluation and trajectories: $OUT_ROOT"
