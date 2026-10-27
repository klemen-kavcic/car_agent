#!/bin/bash
# Five-seed 40M signed-torque experiments.  The policy has only steering and
# one signed motor-torque action; it has no discrete gear/brake action.
# Submit via submit_traffic89_signed_pedal_pipeline.sh.
#
# TRAFFIC_SIGNED_VARIANT=curriculum: 13M recovery lessons, then normal 10-car traffic.
# TRAFFIC_SIGNED_VARIANT=scratch:    normal random 10-car traffic for all 40M steps.
# Final evaluation is intentionally a dependent job, not 40 checkpoint curves.
#SBATCH --job-name=traffic89_signed
#SBATCH --partition=all
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --array=0-4%5
#SBATCH --output=logs/traffic89_signed_%A_%a.out
#SBATCH --error=logs/traffic89_signed_%A_%a.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1

# The submit wrapper changes into its own directory before sbatch, so this
# makes the job portable between mag/car and mag/car1.
ROOT=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
SIF="$ROOT/mlagents.sif"
BUILD="$ROOT/Linux_traffic_training_vehicle_signed_arcade/car.x86_64"
INFO="$ROOT/Linux_traffic_training_vehicle_signed_arcade/build_info.txt"
PREPARE="$ROOT/prepare_multiagent_config.py"
STATS_PATCH="$ROOT/stats_patched.py"
TRAINER_PATCH="$ROOT/trainer_controller_patched.py"
SEED=${SLURM_ARRAY_TASK_ID:?array task required}
VARIANT=${TRAFFIC_SIGNED_VARIANT:?must be curriculum or scratch}
[[ "$VARIANT" =~ ^(curriculum|scratch)$ ]] || { echo 'ERROR: TRAFFIC_SIGNED_VARIANT must be curriculum or scratch' >&2; exit 1; }
TRAIN_CARS=10; NUM_ENVS=6; STEPS=40000000; RECOVERY_STEPS=13000000
SOURCE=${TRAFFIC_TEMPLATE_CONFIG:-"$ROOT/templates/traffic89_laser12_cap3_template.yaml"}
RUN_ID="traffic89_signed_${VARIANT}_${SLURM_ARRAY_JOB_ID}_s${SEED}"
RUN_DIR="$ROOT/results/$RUN_ID"; CONFIG="$ROOT/configs/${RUN_ID}.yaml"

for f in "$SIF" "$BUILD" "$INFO" "$PREPARE" "$STATS_PATCH" "$TRAINER_PATCH" "$SOURCE"; do
    [[ -s "$f" ]] || { echo "ERROR: missing or empty $f" >&2; exit 1; }
done
for line in 'Traffic car pool size: 10;' 'Traffic map source: VoronoiTrain' \
            'Configured Vector Observation Size: 89  [MATCH]' \
            'Actions: 2 continuous, discrete branches [none]' \
            'Traffic control mode: signed-torque arcade' \
            'Traffic signed-pedal direction handling: neutral hold'; do
    grep -Fq "$line" "$INFO" || { echo "ERROR: signed-torque training build lacks '$line'" >&2; exit 1; }
done
mkdir -p "$ROOT/configs" "$ROOT/results" "$ROOT/logs"

PREPARE_ARGS=(--steps "$STEPS" --num-envs "$NUM_ENVS" --summary-freq 100000 \
    --vehicle-distance --signed-pedal-arcade-controls)
if [[ "$VARIANT" == curriculum ]]; then
    PREPARE_ARGS+=(--recovery-curriculum --recovery-curriculum-end-steps "$RECOVERY_STEPS")
fi
apptainer exec "$SIF" python3 "$PREPARE" --source "$SOURCE" --out "$CONFIG" "${PREPARE_ARGS[@]}"

# Each task owns a 64-port block. This lock plus socket check avoids the
# worker/controller startup collisions seen in earlier arrays.
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
trap cleanup EXIT
trap 'cleanup; exit 143' INT TERM

echo "Training $RUN_ID: signed torque; ${VARIANT}; 89 obs; ${NUM_ENVS} environments; ${STEPS} steps; port $PORT_BASE"
apptainer exec \
    --bind "$STATS_PATCH:/usr/local/lib/python3.10/dist-packages/mlagents/trainers/stats.py" \
    --bind "$TRAINER_PATCH:/usr/local/lib/python3.10/dist-packages/mlagents/trainers/trainer_controller.py" \
    "$SIF" xvfb-run -a mlagents-learn "$CONFIG" --env="$BUILD" --run-id="$RUN_ID" \
        --results-dir="$ROOT/results" --no-graphics --torch-device=cpu --num-envs="$NUM_ENVS" \
        --base-port="$PORT_BASE" --seed="$SEED" --env-args -job-worker-count 1 -traffic-car-count "$TRAIN_CARS"
test -s "$RUN_DIR/CarAgent.onnx"
echo "Done: $RUN_ID"
