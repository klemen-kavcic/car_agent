#!/bin/bash
# Five 89-observation, 3 m/s traffic runs.  Each run uses the staged recovery
# lessons for its first 13M samples, then automatically returns to ordinary
# random 10-car VoronoiTrain traffic for the remainder of its 40M budget.
# The existing post-training helper produces the usual 2/5/10-car,
# deterministic/stochastic final evaluation and trajectories.
# Submit: sbatch traffic89_recovery_curriculum_train_eval_hpc.sh
#SBATCH --job-name=traffic89_recovery
#SBATCH --partition=all
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --array=0-4%6
#SBATCH --output=/d/hpc/home/kk42117/mag/car1/logs/traffic89_recovery_%A_%a.out
#SBATCH --error=/d/hpc/home/kk42117/mag/car1/logs/traffic89_recovery_%A_%a.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1

ROOT=/d/hpc/home/kk42117/mag/car1
SIF="$ROOT/mlagents.sif"
PREPARE="$ROOT/prepare_multiagent_config.py"
POST_EVAL="$ROOT/traffic_vehicle_distance_post_train_eval.sh"
SELECTOR="$ROOT/select_multiagent_checkpoint.py"
STATS_PATCH="$ROOT/stats_patched.py"
TRAINER_PATCH="$ROOT/trainer_controller_patched.py"
SEED=${SLURM_ARRAY_TASK_ID:?array task required}
TRAIN_CARS=10; NUM_ENVS=6; STEPS=${TRAFFIC_TRAIN_STEPS:-40000000}
RECOVERY_STEPS=${TRAFFIC_RECOVERY_STEPS:-13000000}
# Default zero is the unshaped reverse curriculum. Set (for example) 0.1 in
# a separate A/B run to add the one-off parked-blocker clearance milestone.
BLOCKER_CLEARANCE_REWARD=${TRAFFIC_RECOVERY_BLOCKER_CLEARANCE_REWARD:-0}
RECOVERY_VARIANT=${TRAFFIC_RECOVERY_VARIANT:-unshaped}
CONTROL_MODE=${TRAFFIC_CONTROL_MODE:-explicit}
[[ "$RECOVERY_VARIANT" =~ ^(unshaped|clearance)$ ]] || {
    echo "ERROR: TRAFFIC_RECOVERY_VARIANT must be unshaped or clearance" >&2; exit 1;
}
[[ "$CONTROL_MODE" =~ ^(explicit|arcade)$ ]] || {
    echo "ERROR: TRAFFIC_CONTROL_MODE must be explicit or arcade" >&2; exit 1;
}
if [[ "$CONTROL_MODE" == arcade ]]; then
    BUILD="$ROOT/Linux_traffic_training_vehicle_arcade/car.x86_64"
    INFO="$ROOT/Linux_traffic_training_vehicle_arcade/build_info.txt"
    EVAL_BUILD="$ROOT/Linux_traffic_all_validation_vehicle_arcade/car.x86_64"
    EVAL_INFO="$ROOT/Linux_traffic_all_validation_vehicle_arcade/build_info.txt"
    PREPARE_CONTROL=(--arcade-reverse-controls)
    RUN_PREFIX="traffic89_recovery_arcade"
else
    BUILD="$ROOT/Linux_traffic_training_vehicle/car.x86_64"
    INFO="$ROOT/Linux_traffic_training_vehicle/build_info.txt"
    EVAL_BUILD="$ROOT/Linux_traffic_all_validation_vehicle/car.x86_64"
    EVAL_INFO="$ROOT/Linux_traffic_all_validation_vehicle/build_info.txt"
    PREPARE_CONTROL=()
    RUN_PREFIX="traffic89_recovery"
fi
SOURCE=${TRAFFIC_TEMPLATE_CONFIG:-"$ROOT/templates/traffic89_laser12_cap3_template.yaml"}
RUN_ID="${RUN_PREFIX}_${RECOVERY_VARIANT}_${SLURM_ARRAY_JOB_ID}_s${SEED}"
RUN_DIR="$ROOT/results/$RUN_ID"; CONFIG="$ROOT/configs/${RUN_ID}.yaml"

for f in "$SIF" "$BUILD" "$INFO" "$EVAL_BUILD" "$EVAL_INFO" "$PREPARE" "$POST_EVAL" "$SELECTOR" "$STATS_PATCH" "$TRAINER_PATCH" "$SOURCE"; do
    [[ -s "$f" ]] || { echo "ERROR: missing or empty $f" >&2; exit 1; }
done
for line in 'Traffic car pool size: 10;' 'Traffic map source: VoronoiTrain' 'Configured Vector Observation Size: 89  [MATCH]' 'Traffic recovery curriculum support:'; do
    grep -Fq "$line" "$INFO" || { echo "ERROR: training build lacks '$line'" >&2; exit 1; }
done
if [[ "$CONTROL_MODE" == arcade ]]; then
    grep -Fq 'Actions: 2 continuous, discrete branches [2]' "$INFO" || {
        echo 'ERROR: arcade training build does not have one [drive/brake] branch' >&2; exit 1;
    }
    grep -Fq 'Actions: 2 continuous, discrete branches [2]' "$EVAL_INFO" || {
        echo 'ERROR: arcade validation build does not have one [drive/brake] branch' >&2; exit 1;
    }
fi
mkdir -p "$ROOT/configs" "$ROOT/results" "$ROOT/logs"

apptainer exec "$SIF" python3 "$PREPARE" --source "$SOURCE" --out "$CONFIG" \
    --steps "$STEPS" --num-envs "$NUM_ENVS" --summary-freq 100000 --vehicle-distance \
    "${PREPARE_CONTROL[@]}" \
    --recovery-curriculum --recovery-curriculum-end-steps "$RECOVERY_STEPS" \
    --recovery-blocker-clearance-reward "$BLOCKER_CLEARANCE_REWARD"

# One 64-port block per task.  The lock and listening-port check prevent the
# worker/controller collision which caused previous array failures.
PORT_BLOCK=64; PORT_FIRST=10000; PORT_SLOT_COUNT=800; PORT_LOCK=
PREFERRED_SLOT=$(((SLURM_ARRAY_JOB_ID * 31 + SEED) % PORT_SLOT_COUNT))
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

echo "Training $RUN_ID: ${RECOVERY_STEPS} recovery + normal 10-car traffic to ${STEPS} steps; 89 obs; ${NUM_ENVS} envs; ${CONTROL_MODE}; ${RECOVERY_VARIANT}; blocker-clearance reward ${BLOCKER_CLEARANCE_REWARD}; port $PORT_BASE"
apptainer exec \
    --bind "$STATS_PATCH:/usr/local/lib/python3.10/dist-packages/mlagents/trainers/stats.py" \
    --bind "$TRAINER_PATCH:/usr/local/lib/python3.10/dist-packages/mlagents/trainers/trainer_controller.py" \
    "$SIF" xvfb-run -a mlagents-learn "$CONFIG" --env="$BUILD" --run-id="$RUN_ID" \
        --results-dir="$ROOT/results" --no-graphics --torch-device=cpu --num-envs="$NUM_ENVS" \
        --base-port="$PORT_BASE" --seed="$SEED" --env-args -job-worker-count 1 -traffic-car-count "$TRAIN_CARS"
test -s "$RUN_DIR/CarAgent.onnx"
python3 "$SELECTOR" --run-dir "$RUN_DIR" --max-steps "$STEPS" --validate-all
if [[ ${TRAFFIC_SKIP_POST_EVAL:-0} != 1 ]]; then
    # This helper runs in a child shell, so explicitly export the shared
    # runtime values it validates before launching Unity evaluation.
    export ROOT SIF PORT_BASE
    export TRAFFIC_EVAL_BUILD="$EVAL_BUILD" TRAFFIC_EVAL_BUILD_INFO="$EVAL_INFO"
    bash "$POST_EVAL" "$RUN_ID" "$RUN_DIR/CarAgent.onnx" "$CONFIG"
else
    echo "Skipping final evaluation/trajectories (TRAFFIC_SKIP_POST_EVAL=1)."
fi
echo "Done: $RUN_ID"
