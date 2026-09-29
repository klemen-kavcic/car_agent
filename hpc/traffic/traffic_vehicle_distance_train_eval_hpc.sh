#!/bin/bash
# Clean 89-observation vehicle-distance experiment, trained from scratch only.
# 2/5/10 cars x five seeds x 40M steps; final held-out evaluation and
# trajectories run in the same Slurm task. Submit with:
#   sbatch traffic_vehicle_distance_train_eval_hpc.sh
#SBATCH --job-name=traffic89_full
#SBATCH --partition=all
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --array=0-14%4
#SBATCH --output=/d/hpc/home/kk42117/mag/car1/logs/traffic89_full_%A_%a.out
#SBATCH --error=/d/hpc/home/kk42117/mag/car1/logs/traffic89_full_%A_%a.err
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
((TASK >= 0 && TASK < 15)) || { echo 'ERROR: task must be 0..14' >&2; exit 1; }

DENSITIES=(2 5 10); ENV_COUNTS=(32 12 6)
GROUP=$((TASK / 5)); SEED=$((TASK % 5)); TRAIN_CARS=${DENSITIES[$GROUP]}; NUM_ENVS=${ENV_COUNTS[$GROUP]}
STEPS=40000000
# This 77-value YAML supplies matched PPO/reward settings only. --vehicle-distance
# adds the two 89-observation environment parameters; no old ONNX/PT is loaded.
SOURCE_CONFIG="$ROOT/results/car_vgrid_19016936_laser12_cap3_s${SEED}/configuration.yaml"
RUN_ID="traffic89_${SLURM_ARRAY_JOB_ID}_scratch_cars${TRAIN_CARS}_s${SEED}"
RUN_DIR="$ROOT/results/$RUN_ID"; CONFIG="$ROOT/configs/${RUN_ID}.yaml"

for f in "$SIF" "$BUILD" "$INFO" "$PREPARE" "$POST_EVAL" "$SELECTOR" "$STATS_PATCH" "$TRAINER_PATCH" "$SOURCE_CONFIG"; do
    [[ -s "$f" ]] || { echo "ERROR: missing or empty $f" >&2; exit 1; }
done
for line in 'Traffic car pool size: 10;' 'Traffic map source: VoronoiTrain' 'Configured Vector Observation Size: 89  [MATCH]'; do
    grep -Fq "$line" "$INFO" || { echo "ERROR: 89 training build lacks '$line'" >&2; exit 1; }
done

mkdir -p "$ROOT/logs" "$ROOT/configs" "$ROOT/results"
apptainer exec "$SIF" python3 "$PREPARE" --source "$SOURCE_CONFIG" --out "$CONFIG" \
    --steps "$STEPS" --num-envs "$NUM_ENVS" --vehicle-distance

# Reserve a port block that is both locked by this workflow and actually unused
# on the allocated node. This prevents the collision that broke job 19413469.
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

echo "Training $RUN_ID: scratch, seed=$SEED, cars=$TRAIN_CARS, envs=$NUM_ENVS, steps=$STEPS, port=$PORT_BASE"
apptainer exec \
    --bind "$STATS_PATCH:/usr/local/lib/python3.10/dist-packages/mlagents/trainers/stats.py" \
    --bind "$TRAINER_PATCH:/usr/local/lib/python3.10/dist-packages/mlagents/trainers/trainer_controller.py" \
    "$SIF" xvfb-run -a mlagents-learn "$CONFIG" \
        --env="$BUILD" --run-id="$RUN_ID" --results-dir="$ROOT/results" \
        --no-graphics --torch-device=cpu --num-envs="$NUM_ENVS" --base-port="$PORT_BASE" --seed="$SEED" \
        --env-args -job-worker-count 1 -traffic-car-count "$TRAIN_CARS"
test -s "$RUN_DIR/CarAgent.onnx"
python3 "$SELECTOR" --run-dir "$RUN_DIR" --max-steps "$STEPS" --validate-all
bash "$POST_EVAL" "$RUN_ID" "$RUN_DIR/CarAgent.onnx" "$CONFIG"
echo "Done: $RUN_ID"
