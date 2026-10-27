#!/bin/bash
# Two isolated 89-observation traffic curricula in one 10-task array:
#   task 0--4  smooth per-episode 2--10-car density distribution
#   task 5--9  fixed 10 cars, smooth short-to-long route-length curriculum
# Both are scratch 40M PPO runs with five independent seeds, then the usual
# 2/5/10-car deterministic + stochastic final evaluation and trajectories.
# Submit: sbatch traffic89_smooth_curricula_train_eval_hpc.sh
#SBATCH --job-name=traffic89_smooth
#SBATCH --partition=all
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --array=0-9
#SBATCH --output=/d/hpc/home/kk42117/mag/car1/logs/traffic89_smooth_%A_%a.out
#SBATCH --error=/d/hpc/home/kk42117/mag/car1/logs/traffic89_smooth_%A_%a.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1

ROOT=/d/hpc/home/kk42117/mag/car1
SIF="$ROOT/mlagents.sif"
BUILD="$ROOT/Linux_traffic_training_vehicle/car.x86_64"
INFO="$ROOT/Linux_traffic_training_vehicle/build_info.txt"
PREPARE="$ROOT/prepare_multiagent_config.py"
POST_EVAL="$ROOT/traffic_vehicle_distance_post_train_eval.sh"
SELECTOR="$ROOT/select_multiagent_checkpoint.py"
STATS_PATCH="$ROOT/stats_patched.py"
TRAINER_PATCH="$ROOT/trainer_controller_patched.py"
TASK=${SLURM_ARRAY_TASK_ID:?array task required}
TRAIN_CARS=10; NUM_ENVS=6; STEPS=${TRAFFIC_TRAIN_STEPS:-40000000}
SOURCE=${TRAFFIC_TEMPLATE_CONFIG:-"$ROOT/results/car_vgrid_19016936_laser12_cap3_s0/configuration.yaml"}

if (( TASK < 5 )); then
    SEED=$TASK; FAMILY=smooth_count
    CURRICULUM_ARG=(--smooth-car-count-curriculum)
else
    SEED=$((TASK - 5)); FAMILY=smooth_route
    CURRICULUM_ARG=(--smooth-route-length-curriculum)
fi
RUN_ID="traffic89_${FAMILY}_${SLURM_ARRAY_JOB_ID}_s${SEED}"
RUN_DIR="$ROOT/results/$RUN_ID"; CONFIG="$ROOT/configs/${RUN_ID}.yaml"

for f in "$SIF" "$BUILD" "$INFO" "$PREPARE" "$POST_EVAL" "$SELECTOR" "$STATS_PATCH" "$TRAINER_PATCH" "$SOURCE"; do
    [[ -s "$f" ]] || { echo "ERROR: missing or empty $f" >&2; exit 1; }
done
for line in 'Traffic car pool size: 10;' 'Traffic map source: VoronoiTrain' 'Configured Vector Observation Size: 89  [MATCH]'; do
    grep -Fq "$line" "$INFO" || { echo "ERROR: training build lacks '$line'" >&2; exit 1; }
done

mkdir -p "$ROOT/configs" "$ROOT/results" "$ROOT/logs"
apptainer exec "$SIF" python3 "$PREPARE" --source "$SOURCE" --out "$CONFIG" \
    --steps "$STEPS" --num-envs "$NUM_ENVS" --vehicle-distance "${CURRICULUM_ARG[@]}"

# The traffic manager needs all ten pooled cars available at launch. The
# smooth-count curriculum activates a sampled 2--10-car prefix per episode.
PORT_BLOCK=64; PORT_FIRST=10000; PORT_SLOT_COUNT=800; PORT_LOCK=
PREFERRED_SLOT=$(((SLURM_ARRAY_JOB_ID * 31 + TASK) % PORT_SLOT_COUNT))
for ((ATTEMPT=0; ATTEMPT<PORT_SLOT_COUNT; ATTEMPT++)); do
    SLOT=$(((PREFERRED_SLOT + ATTEMPT) % PORT_SLOT_COUNT))
    CANDIDATE_LOCK="/tmp/car_mlagents_portblock_${SLOT}.lock"
    if mkdir "$CANDIDATE_LOCK" 2>/dev/null; then
        CANDIDATE_BASE=$((PORT_FIRST + SLOT * PORT_BLOCK)); BLOCK_FREE=1
        if command -v ss >/dev/null 2>&1; then
            for ((OFFSET=0; OFFSET<PORT_BLOCK; OFFSET++)); do
                if ss -H -ltn "sport = :$((CANDIDATE_BASE + OFFSET))" | grep -q .; then BLOCK_FREE=0; break; fi
            done
        fi
        if ((BLOCK_FREE)); then PORT_LOCK="$CANDIDATE_LOCK"; PORT_BASE="$CANDIDATE_BASE"; break; fi
        rmdir "$CANDIDATE_LOCK" 2>/dev/null || true
    fi
done
[[ -n "$PORT_LOCK" ]] || { echo 'ERROR: no free ML-Agents port block' >&2; exit 1; }
cleanup(){ rmdir "$PORT_LOCK" 2>/dev/null || true; }
trap cleanup EXIT; trap 'cleanup; exit 143' INT TERM

echo "Training $RUN_ID: $FAMILY curriculum, 89 observations, 6 envs, 40M steps, port $PORT_BASE"
apptainer exec \
    --bind "$STATS_PATCH:/usr/local/lib/python3.10/dist-packages/mlagents/trainers/stats.py" \
    --bind "$TRAINER_PATCH:/usr/local/lib/python3.10/dist-packages/mlagents/trainers/trainer_controller.py" \
    "$SIF" xvfb-run -a mlagents-learn "$CONFIG" --env="$BUILD" --run-id="$RUN_ID" \
        --results-dir="$ROOT/results" --no-graphics --torch-device=cpu --num-envs="$NUM_ENVS" \
        --base-port="$PORT_BASE" --seed="$SEED" --env-args -job-worker-count 1 -traffic-car-count "$TRAIN_CARS"
test -s "$RUN_DIR/CarAgent.onnx"
python3 "$SELECTOR" --run-dir "$RUN_DIR" --max-steps "$STEPS" --validate-all
if [[ ${TRAFFIC_SKIP_POST_EVAL:-0} != 1 ]]; then
    ROOT="$ROOT" SIF="$SIF" PORT_BASE="$PORT_BASE" bash "$POST_EVAL" "$RUN_ID" "$RUN_DIR/CarAgent.onnx" "$CONFIG"
else
    echo "Skipping final evaluation/trajectories (TRAFFIC_SKIP_POST_EVAL=1)."
fi
echo "Done: $RUN_ID"
