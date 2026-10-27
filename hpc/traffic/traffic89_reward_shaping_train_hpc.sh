#!/bin/bash
# Two isolated 40M reward-shaping experiments on the normal explicit-gear,
# 89-observation, 10-car traffic setup. Select one via TRAFFIC_REWARD_VARIANT:
#   idle        escalating low-speed penalty only
#   exponential exponentially weighted checkpoint rewards only
# Final evaluation and 5M curves are submitted separately by the wrapper.
#SBATCH --job-name=traffic89_reward
#SBATCH --partition=all
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --array=0-4%5
#SBATCH --output=/d/hpc/home/kk42117/mag/car1/logs/traffic89_reward_%A_%a.out
#SBATCH --error=/d/hpc/home/kk42117/mag/car1/logs/traffic89_reward_%A_%a.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1

ROOT=/d/hpc/home/kk42117/mag/car1
SIF="$ROOT/mlagents.sif"
BUILD="$ROOT/Linux_traffic_training_vehicle_reward/car.x86_64"
INFO="$ROOT/Linux_traffic_training_vehicle_reward/build_info.txt"
PREPARE="$ROOT/prepare_multiagent_config.py"
STATS_PATCH="$ROOT/stats_patched.py"
TRAINER_PATCH="$ROOT/trainer_controller_patched.py"
SEED=${SLURM_ARRAY_TASK_ID:?array task required}
VARIANT=${TRAFFIC_REWARD_VARIANT:?must be idle or exponential}
[[ "$VARIANT" =~ ^(idle|exponential)$ ]] || { echo 'ERROR: TRAFFIC_REWARD_VARIANT must be idle or exponential' >&2; exit 1; }
TRAIN_CARS=10; NUM_ENVS=6; STEPS=40000000
SOURCE=${TRAFFIC_TEMPLATE_CONFIG:-"$ROOT/templates/traffic89_laser12_cap3_template.yaml"}
RUN_ID="traffic89_reward_${VARIANT}_${SLURM_ARRAY_JOB_ID}_s${SEED}"
RUN_DIR="$ROOT/results/$RUN_ID"; CONFIG="$ROOT/configs/${RUN_ID}.yaml"

for f in "$SIF" "$BUILD" "$INFO" "$PREPARE" "$STATS_PATCH" "$TRAINER_PATCH" "$SOURCE"; do
    [[ -s "$f" ]] || { echo "ERROR: missing or empty $f" >&2; exit 1; }
done
for line in 'Traffic car pool size: 10;' 'Traffic map source: VoronoiTrain' \
            'Configured Vector Observation Size: 89  [MATCH]' 'Actions: 2 continuous, discrete branches [2,2]' \
            'Traffic reward-shaping support:'; do
    grep -Fq "$line" "$INFO" || { echo "ERROR: normal training build lacks '$line'" >&2; exit 1; }
done
mkdir -p "$ROOT/configs" "$ROOT/results" "$ROOT/logs"

PREPARE_ARGS=(--steps "$STEPS" --num-envs "$NUM_ENVS" --summary-freq 100000 --vehicle-distance)
if [[ "$VARIANT" == idle ]]; then
    # 0.15 m/s is 5% of the 3 m/s cap: ignores tiny jitter, but not careful driving.
    PREPARE_ARGS+=(--idle-penalty --idle-speed-threshold 0.15 --idle-grace-steps 20 \
        --idle-penalty-base -0.02 --idle-penalty-growth 0.05)
else
    # Equal rewards formerly totalled 100 * 0.02 = 2.0. Preserve that budget.
    PREPARE_ARGS+=(--exponential-checkpoint-rewards \
        --checkpoint-exponential-total-reward 2.0 --checkpoint-exponential-beta 0.04)
fi
apptainer exec "$SIF" python3 "$PREPARE" --source "$SOURCE" --out "$CONFIG" "${PREPARE_ARGS[@]}"

PORT_BLOCK=64; PORT_FIRST=10000; PORT_SLOT_COUNT=800; PORT_LOCK=
PREFERRED_SLOT=$(((SLURM_ARRAY_JOB_ID * 31 + SEED) % PORT_SLOT_COUNT))
for ((ATTEMPT=0; ATTEMPT<PORT_SLOT_COUNT; ATTEMPT++)); do
    SLOT=$(((PREFERRED_SLOT + ATTEMPT) % PORT_SLOT_COUNT)); CANDIDATE_LOCK="/tmp/car_mlagents_portblock_${SLOT}.lock"
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
trap cleanup EXIT
trap 'cleanup; exit 143' INT TERM

echo "Training $RUN_ID: ${VARIANT} reward shaping; 89 obs; explicit controls; ${NUM_ENVS} environments; ${STEPS} steps; port $PORT_BASE"
apptainer exec \
    --bind "$STATS_PATCH:/usr/local/lib/python3.10/dist-packages/mlagents/trainers/stats.py" \
    --bind "$TRAINER_PATCH:/usr/local/lib/python3.10/dist-packages/mlagents/trainers/trainer_controller.py" \
    "$SIF" xvfb-run -a mlagents-learn "$CONFIG" --env="$BUILD" --run-id="$RUN_ID" \
        --results-dir="$ROOT/results" --no-graphics --torch-device=cpu --num-envs="$NUM_ENVS" \
        --base-port="$PORT_BASE" --seed="$SEED" --env-args -job-worker-count 1 -traffic-car-count "$TRAIN_CARS"
test -s "$RUN_DIR/CarAgent.onnx"
echo "Done: $RUN_ID"
