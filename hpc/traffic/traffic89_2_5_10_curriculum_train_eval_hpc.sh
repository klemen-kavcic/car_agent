#!/bin/bash
# Controlled 89-observation traffic curriculum: the same PPO run trains for
# 10M steps with 2 cars, resumes for 10M with 5 cars, then resumes for 20M
# with 10 cars. Five scratch seeds; final evaluation matches prior experiments.
# Submit: sbatch traffic89_2_5_10_curriculum_train_eval_hpc.sh
#SBATCH --job-name=traffic89_2to5to10
#SBATCH --partition=all
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --array=0-4%3
#SBATCH --output=/d/hpc/home/kk42117/mag/car1/logs/traffic89_2to5to10_%A_%a.out
#SBATCH --error=/d/hpc/home/kk42117/mag/car1/logs/traffic89_2to5to10_%A_%a.err
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
SEED=${SLURM_ARRAY_TASK_ID:?array task required}
RUN_ID="traffic89_2to5to10_${SLURM_ARRAY_JOB_ID}_s${SEED}"
RUN_DIR="$ROOT/results/$RUN_ID"; CONFIG="$ROOT/configs/${RUN_ID}.yaml"
# This is scratch training. Reuse one exact archived cap3 YAML as the common
# PPO/reward template; --seed below still makes all five runs independent.
SOURCE=${TRAFFIC_TEMPLATE_CONFIG:-"$ROOT/results/car_vgrid_19016936_laser12_cap3_s0/configuration.yaml"}

for f in "$SIF" "$BUILD" "$INFO" "$PREPARE" "$POST_EVAL" "$SELECTOR" "$STATS_PATCH" "$TRAINER_PATCH" "$SOURCE"; do
    [[ -s "$f" ]] || { echo "ERROR: missing or empty $f" >&2; exit 1; }
done
for line in 'Traffic car pool size: 10;' 'Traffic map source: VoronoiTrain' 'Configured Vector Observation Size: 89  [MATCH]'; do
    grep -Fq "$line" "$INFO" || { echo "ERROR: training build lacks '$line'" >&2; exit 1; }
done

mkdir -p "$ROOT/configs" "$ROOT/results" "$ROOT/logs"
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

run_phase() {
    local label=$1 target_steps=$2 cars=$3 envs=$4 resume=$5
    apptainer exec "$SIF" python3 "$PREPARE" --source "$SOURCE" --out "$CONFIG" \
        --steps "$target_steps" --num-envs "$envs" --vehicle-distance
    echo "Phase $label: $RUN_ID, cumulative $target_steps steps, $cars cars, $envs environments"
    local resume_arg=()
    [[ "$resume" == 1 ]] && resume_arg=(--resume)
    apptainer exec \
        --bind "$STATS_PATCH:/usr/local/lib/python3.10/dist-packages/mlagents/trainers/stats.py" \
        --bind "$TRAINER_PATCH:/usr/local/lib/python3.10/dist-packages/mlagents/trainers/trainer_controller.py" \
        "$SIF" xvfb-run -a mlagents-learn "$CONFIG" --env="$BUILD" --run-id="$RUN_ID" \
            --results-dir="$ROOT/results" --no-graphics --torch-device=cpu --num-envs="$envs" \
            --base-port="$PORT_BASE" --seed="$SEED" "${resume_arg[@]}" \
            --env-args -job-worker-count 1 -traffic-car-count "$cars"
}

run_phase 2cars 10000000 2 32 0
run_phase 5cars 20000000 5 12 1
run_phase 10cars 40000000 10 6 1
test -s "$RUN_DIR/CarAgent.onnx"
python3 "$SELECTOR" --run-dir "$RUN_DIR" --max-steps 40000000 --validate-all
ROOT="$ROOT" SIF="$SIF" PORT_BASE="$PORT_BASE" bash "$POST_EVAL" "$RUN_ID" "$RUN_DIR/CarAgent.onnx" "$CONFIG"
echo "Done: $RUN_ID"
