#!/bin/bash
# Short, five-seed diagnostic training run for the recovery curriculum only.
# It ends exactly at 7.5M trainer steps, i.e. after the one-car, parked-car,
# two-car and four-car recovery stages, before ordinary Voronoi traffic begins.
# It deliberately has no post-training evaluation: TensorBoard summaries every
# 10k steps give a detailed diagnostic trace for every short lesson.
# Submit: sbatch traffic89_recovery75m_probe_hpc.sh
#SBATCH --job-name=traffic89_rec75m
#SBATCH --partition=all
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --array=0-4%6
#SBATCH --output=/d/hpc/home/kk42117/mag/car1/logs/traffic89_rec75m_%A_%a.out
#SBATCH --error=/d/hpc/home/kk42117/mag/car1/logs/traffic89_rec75m_%A_%a.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1

ROOT=/d/hpc/home/kk42117/mag/car1
SIF="$ROOT/mlagents.sif"
BUILD="$ROOT/Linux_traffic_training_vehicle/car.x86_64"
INFO="$ROOT/Linux_traffic_training_vehicle/build_info.txt"
PREPARE="$ROOT/prepare_multiagent_config.py"
STATS_PATCH="$ROOT/stats_patched.py"
TRAINER_PATCH="$ROOT/trainer_controller_patched.py"
SEED=${SLURM_ARRAY_TASK_ID:?array task required}
STEPS=${RECOVERY_PROBE_STEPS:-7500000}; NUM_ENVS=6; TRAIN_CARS=10
PROBE_LABEL=${RECOVERY_PROBE_LABEL:-recovery75m}
SOURCE=${TRAFFIC_TEMPLATE_CONFIG:-"$ROOT/templates/traffic89_laser12_cap3_template.yaml"}
RUN_ID="traffic89_${PROBE_LABEL}_${SLURM_ARRAY_JOB_ID}_s${SEED}"
RUN_DIR="$ROOT/results/$RUN_ID"; CONFIG="$ROOT/configs/${RUN_ID}.yaml"

for f in "$SIF" "$BUILD" "$INFO" "$PREPARE" "$STATS_PATCH" "$TRAINER_PATCH" "$SOURCE"; do
    [[ -s "$f" ]] || { echo "ERROR: missing or empty $f" >&2; exit 1; }
done
for line in 'Traffic car pool size: 10;' 'Traffic map source: VoronoiTrain' 'Configured Vector Observation Size: 89  [MATCH]' 'Traffic recovery curriculum support:'; do
    grep -Fq "$line" "$INFO" || { echo "ERROR: training build lacks '$line'" >&2; exit 1; }
done
mkdir -p "$ROOT/configs" "$ROOT/results" "$ROOT/logs"

apptainer exec "$SIF" python3 "$PREPARE" --source "$SOURCE" --out "$CONFIG" \
    --steps "$STEPS" --num-envs "$NUM_ENVS" --summary-freq 10000 --vehicle-distance \
    --recovery-curriculum --recovery-curriculum-end-steps "$STEPS"

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

echo "Recovery probe $RUN_ID: ${STEPS} steps only, 89 obs, ${NUM_ENVS} environments, 10k summaries, port $PORT_BASE"
apptainer exec \
    --bind "$STATS_PATCH:/usr/local/lib/python3.10/dist-packages/mlagents/trainers/stats.py" \
    --bind "$TRAINER_PATCH:/usr/local/lib/python3.10/dist-packages/mlagents/trainers/trainer_controller.py" \
    "$SIF" xvfb-run -a mlagents-learn "$CONFIG" --env="$BUILD" --run-id="$RUN_ID" \
        --results-dir="$ROOT/results" --no-graphics --torch-device=cpu --num-envs="$NUM_ENVS" \
        --base-port="$PORT_BASE" --seed="$SEED" --env-args -job-worker-count 1 -traffic-car-count "$TRAIN_CARS"
test -s "$RUN_DIR/CarAgent.onnx"
echo "Done: $RUN_ID"
