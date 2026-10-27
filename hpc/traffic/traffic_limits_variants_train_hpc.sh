#!/bin/bash
# Three matched all-outcomes-removal speed/reverse-limit experiments:
#   time90_limits_128       : 90 observations, 128-unit layers
#   dense162_limits_128     : 162 observations, 128-unit layers
#   dense162_limits_256     : 162 observations, 256-unit layers
# Submit through submit_traffic_limits_variants_pipeline.sh.
#SBATCH --job-name=traffic_limits
#SBATCH --partition=all
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --array=0-4%5
#SBATCH --output=logs/traffic_limits_%A_%a.out
#SBATCH --error=logs/traffic_limits_%A_%a.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1

ROOT=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
SIF="$ROOT/mlagents.sif"; PREPARE="$ROOT/prepare_multiagent_config.py"
STATS_PATCH="$ROOT/stats_patched.py"; TRAINER_PATCH="$ROOT/trainer_controller_patched.py"
CPO_TRAINER_PATCH="$ROOT/ppo_constrained_trainer.py"; CPO_OPTIMIZER_PATCH="$ROOT/ppo_constrained_optimizer.py"
SOURCE=${TRAFFIC_TEMPLATE_CONFIG:-"$ROOT/templates/traffic89_laser12_cap3_template.yaml"}
VARIANT=${TRAFFIC_LIMITS_VARIANT:?submit wrapper must provide TRAFFIC_LIMITS_VARIANT}
SEED=${SLURM_ARRAY_TASK_ID:?array task required}
STEPS=${TRAFFIC_LIMITS_STEPS:-40000000}; TRAIN_CARS=10; NUM_ENVS=6

case "$VARIANT" in
  time90_limits_128)
    RUN_PREFIX=traffic90_time_remove_finished_limits; OBS=90; HIDDEN=128
    BUILD="$ROOT/Linux_traffic_training_vehicle_signed_arcade_time/car.x86_64"
    INFO="$ROOT/Linux_traffic_training_vehicle_signed_arcade_time/build_info.txt" ;;
  dense162_limits_128)
    RUN_PREFIX=traffic162_dense_time_remove_finished_limits128; OBS=162; HIDDEN=128
    BUILD="$ROOT/Linux_traffic_training_vehicle_signed_arcade_time_dense/car.x86_64"
    INFO="$ROOT/Linux_traffic_training_vehicle_signed_arcade_time_dense/build_info.txt" ;;
  dense162_limits_256)
    RUN_PREFIX=traffic162_dense_time_remove_finished_limits256; OBS=162; HIDDEN=256
    BUILD="$ROOT/Linux_traffic_training_vehicle_signed_arcade_time_dense/car.x86_64"
    INFO="$ROOT/Linux_traffic_training_vehicle_signed_arcade_time_dense/build_info.txt" ;;
  dense162_limits_256_stable)
    RUN_PREFIX=traffic162_dense_time_remove_finished_limits256_stable; OBS=162; HIDDEN=256
    BUILD="$ROOT/Linux_traffic_training_vehicle_signed_arcade_time_dense/car.x86_64"
    INFO="$ROOT/Linux_traffic_training_vehicle_signed_arcade_time_dense/build_info.txt" ;;
  dense162_limits_256_middle)
    RUN_PREFIX=traffic162_dense_time_remove_finished_limits256_middle; OBS=162; HIDDEN=256
    BUILD="$ROOT/Linux_traffic_training_vehicle_signed_arcade_time_dense/car.x86_64"
    INFO="$ROOT/Linux_traffic_training_vehicle_signed_arcade_time_dense/build_info.txt" ;;
  time90_limits_lagrangian)
    RUN_PREFIX=traffic90_time_remove_finished_limits_lagrangian; OBS=90; HIDDEN=128
    BUILD="$ROOT/Linux_traffic_training_vehicle_signed_arcade_time/car.x86_64"
    INFO="$ROOT/Linux_traffic_training_vehicle_signed_arcade_time/build_info.txt" ;;
  time90_limits_constrained_ppo)
    RUN_PREFIX=traffic90_time_remove_finished_limits_cpo; OBS=90; HIDDEN=128
    BUILD="$ROOT/Linux_traffic_training_vehicle_signed_arcade_time/car.x86_64"
    INFO="$ROOT/Linux_traffic_training_vehicle_signed_arcade_time/build_info.txt" ;;
  time90_limits_cpo_t12|time90_limits_cpo_t10|time90_limits_cpo_t08|time90_limits_cpo_t05)
    RUN_PREFIX="traffic90_time_remove_finished_limits_cpo_${VARIANT##*_t}"; OBS=90; HIDDEN=128
    BUILD="$ROOT/Linux_traffic_training_vehicle_signed_arcade_time/car.x86_64"
    INFO="$ROOT/Linux_traffic_training_vehicle_signed_arcade_time/build_info.txt" ;;
  time90_limits_60)
    RUN_PREFIX=traffic90_time_remove_finished_limits_60; OBS=90; HIDDEN=128
    BUILD="$ROOT/Linux_traffic_training_vehicle_signed_arcade_time/car.x86_64"
    INFO="$ROOT/Linux_traffic_training_vehicle_signed_arcade_time/build_info.txt" ;;
  time90_limits_cpo_t10_60|time90_limits_cpo_t08_60|time90_limits_cpo_t10_ttc_40|time90_limits_cpo_t10_ttc_60|time90_limits_cpo_t08_ttc_40|time90_limits_cpo_t08_ttc_60)
    RUN_PREFIX="traffic90_time_remove_finished_limits_${VARIANT#time90_limits_}"; OBS=90; HIDDEN=128
    BUILD="$ROOT/Linux_traffic_training_vehicle_signed_arcade_time/car.x86_64"
    INFO="$ROOT/Linux_traffic_training_vehicle_signed_arcade_time/build_info.txt" ;;
  time90_limits_top5_40)
    RUN_PREFIX=traffic90_time_remove_finished_limits_top5_40; OBS=90; HIDDEN=128
    BUILD="$ROOT/Linux_traffic_training_vehicle_signed_arcade_time/car.x86_64"
    INFO="$ROOT/Linux_traffic_training_vehicle_signed_arcade_time/build_info.txt" ;;
  time90_limits_cpo_smoke)
    RUN_PREFIX=traffic90_time_remove_finished_limits_cpo_smoke; OBS=90; HIDDEN=128
    BUILD="$ROOT/Linux_traffic_training_vehicle_signed_arcade_time/car.x86_64"
    INFO="$ROOT/Linux_traffic_training_vehicle_signed_arcade_time/build_info.txt" ;;
  time90_limits_cpo_marker_smoke)
    RUN_PREFIX=traffic90_time_remove_finished_limits_cpo_marker_smoke; OBS=90; HIDDEN=128
    BUILD="$ROOT/Linux_traffic_training_vehicle_signed_arcade_time/car.x86_64"
    INFO="$ROOT/Linux_traffic_training_vehicle_signed_arcade_time/build_info.txt" ;;
  *) echo "ERROR: unknown TRAFFIC_LIMITS_VARIANT=$VARIANT" >&2; exit 2 ;;
esac
RUN_ID="${RUN_PREFIX}_${SLURM_ARRAY_JOB_ID}_s${SEED}"
RUN_DIR="$ROOT/results/$RUN_ID"; CONFIG="$ROOT/configs/${RUN_ID}.yaml"

for f in "$SIF" "$BUILD" "$INFO" "$PREPARE" "$STATS_PATCH" "$TRAINER_PATCH" "$SOURCE"; do
    [[ -s "$f" ]] || { echo "ERROR: missing or empty $f" >&2; exit 1; }
done
for line in 'Traffic car pool size: 10;' 'Traffic map source: VoronoiTrain' \
            'Grid: 66x66, cell size 1.5m, mapSource Voronoi' \
            "Configured Vector Observation Size: ${OBS}  [MATCH]" \
            'includeRemainingStepsObservation: True' \
            'Actions: 2 continuous, discrete branches [none]' \
            'Traffic control mode: signed-torque arcade' \
            'Traffic signed-pedal direction handling: neutral hold'; do
    grep -Fq "$line" "$INFO" || { echo "ERROR: incompatible build lacks '$line'" >&2; exit 1; }
done
mkdir -p "$ROOT/configs" "$ROOT/results" "$ROOT/logs"

PPO_STABILITY_ARGS=()
if [[ "$VARIANT" == "dense162_limits_256_stable" ]]; then
    # Matched 162-observation stability test: lower update magnitude and a
    # four-times-larger rollout, while leaving environment/reward settings intact.
    PPO_STABILITY_ARGS=(--learning-rate 0.00015 --buffer-size 32768 --batch-size 4096)
elif [[ "$VARIANT" == "dense162_limits_256_middle" ]]; then
    # Middle ground: twice the rollout/minibatch and a moderately smaller
    # learning rate, avoiding the fourfold rollout that slowed 40M learning.
    PPO_STABILITY_ARGS=(--learning-rate 0.0002 --buffer-size 16384 --batch-size 2048)
fi

LAGRANGIAN_ARGS=()
if [[ "$VARIANT" == "time90_limits_lagrangian" ]]; then
    # Preserve the baseline -10 collision penalty at activation, then gently
    # adapt it to pursue <=8% VehicleCollision without rewarding timeouts.
    LAGRANGIAN_ARGS=(--crash-lagrangian --crash-lagrangian-target 0.08 \
        --crash-lagrangian-init 10 --crash-lagrangian-lr 1 --crash-lagrangian-max 20)
fi

CPO_PREPARE_ARGS=()
CPO_BINDS=()
CPO_ENV=()
TTC_ARGS=()
SPEED_CAP_ARGS=()
case "$VARIANT" in
  time90_limits_cpo_t10_ttc_40|time90_limits_cpo_t10_ttc_60|time90_limits_cpo_t08_ttc_40|time90_limits_cpo_t08_ttc_60)
    # Small quadratic penalty only for a car closing on the nearest vehicle
    # directly ahead with TTC < 1.5 seconds. CPO still receives only the
    # terminal binary VehicleCollision cost.
    TTC_ARGS=(--vehicle-ttc-risk-penalty -0.005 --vehicle-ttc-risk-threshold-s 1.5 \
              --vehicle-ttc-risk-min-closing-speed-ms 0.25) ;;
esac
if [[ "$VARIANT" == "time90_limits_top5_40" ]]; then
    SPEED_CAP_ARGS=(--vehicle-speed-cap-ms 5.0)
fi
if [[ "$VARIANT" == "time90_limits_constrained_ppo" || "$VARIANT" == time90_limits_cpo_t* || "$VARIANT" == "time90_limits_cpo_smoke" || "$VARIANT" == "time90_limits_cpo_marker_smoke" ]]; then
    # True constrained PPO: preserve the ordinary task reward and attach a
    # separate binary VehicleCollision cost / cost critic / dual variable.
    CPO_PREPARE_ARGS=(--constrained-ppo)
    for f in "$CPO_TRAINER_PATCH" "$CPO_OPTIMIZER_PATCH"; do
        [[ -s "$f" ]] || { echo "ERROR: missing or empty $f" >&2; exit 1; }
    done
    CPO_BINDS=(
        --bind "$CPO_TRAINER_PATCH:/usr/local/lib/python3.10/dist-packages/mlagents/trainers/ppo/trainer.py"
        --bind "$CPO_OPTIMIZER_PATCH:/usr/local/lib/python3.10/dist-packages/mlagents/trainers/ppo/optimizer_torch.py"
    )
    case "$VARIANT" in
      time90_limits_cpo_t12) CPO_TARGET=0.12 ;;
      time90_limits_cpo_t10|time90_limits_cpo_t10_60|time90_limits_cpo_t10_ttc_40|time90_limits_cpo_t10_ttc_60) CPO_TARGET=0.10 ;;
      time90_limits_cpo_t08|time90_limits_cpo_t08_60|time90_limits_cpo_t08_ttc_40|time90_limits_cpo_t08_ttc_60|time90_limits_constrained_ppo) CPO_TARGET=0.08 ;;
      time90_limits_cpo_t05) CPO_TARGET=0.05 ;;
      time90_limits_cpo_smoke) CPO_TARGET=0.00 ;;
      time90_limits_cpo_marker_smoke) CPO_TARGET=0.00 ;;
      *) echo "ERROR: unsupported constrained-PPO variant $VARIANT" >&2; exit 2 ;;
    esac
    CPO_LAMBDA_MAX=10
    # The prior 8% run approached the old ceiling; allow continued dual ascent.
    if [[ "$CPO_TARGET" == "0.08" ]]; then CPO_LAMBDA_MAX=15; fi
    CPO_ENV=(CPO_TARGET_CRASH_RATE="$CPO_TARGET" CPO_LAMBDA_INIT=0 CPO_DUAL_LR=0.20 CPO_LAMBDA_MAX="$CPO_LAMBDA_MAX")
    if [[ "$VARIANT" == "time90_limits_cpo_smoke" ]]; then
        CPO_ENV+=(CPO_SMOKE_FORCE_COST=1)
    fi
fi

apptainer exec "$SIF" python3 "$PREPARE" --source "$SOURCE" --out "$CONFIG" \
    --steps "$STEPS" --num-envs "$NUM_ENVS" --summary-freq 100000 --hidden-units "$HIDDEN" \
    "${PPO_STABILITY_ARGS[@]}" \
    "${LAGRANGIAN_ARGS[@]}" \
    "${CPO_PREPARE_ARGS[@]}" \
    "${TTC_ARGS[@]}" "${SPEED_CAP_ARGS[@]}" \
    --vehicle-distance --signed-pedal-arcade-controls --remove-finished-cars \
    --speed-limited-speed-fraction 0.10 --signed-pedal-reverse-torque-fraction 0.25

# Parse the generated configuration through exactly the same plugin registry
# and bind-mounted PPO modules that mlagents-learn will use.  Calling
# RunOptions.from_dict without registering trainer plugins first is *not* a
# valid preflight: it falsely says that the normal built-in "ppo" trainer is
# invalid.  This turns an accidental configuration/module mismatch into an
# immediate, readable Slurm error before any Unity worker is launched.
if [[ "$VARIANT" == "time90_limits_constrained_ppo" || "$VARIANT" == time90_limits_cpo_t* || "$VARIANT" == "time90_limits_cpo_smoke" || "$VARIANT" == "time90_limits_cpo_marker_smoke" ]]; then
    apptainer exec "${CPO_BINDS[@]}" "$SIF" python3 - "$CONFIG" <<'PY'
import sys
import yaml

from mlagents.plugins.trainer_type import register_trainer_plugins
register_trainer_plugins()
from mlagents.trainers.settings import RunOptions

with open(sys.argv[1], encoding="utf-8") as config_file:
    RunOptions.from_dict(yaml.safe_load(config_file))
print("Constrained-PPO configuration parse: OK")
PY
fi

# Reserve one non-overlapping 64-port block for six Unity workers.
PORT_BLOCK=64; PORT_FIRST=10000; PORT_SLOT_COUNT=800; PORT_LOCK=
PREFERRED_SLOT=$(((SLURM_ARRAY_JOB_ID * 31 + SEED * 17) % PORT_SLOT_COUNT))
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

echo "Training $RUN_ID: $VARIANT; ${OBS} observations; hidden=${HIDDEN}; speed-limit=10%; reverse torque=25%; remove all finished cars; TTC=${TTC_ARGS[*]:-off}; cap=${SPEED_CAP_ARGS[*]:-3 m/s}; ${PPO_STABILITY_ARGS[*]:-source PPO settings}; ${LAGRANGIAN_ARGS[*]:-no crash Lagrangian}; port $PORT_BASE"
apptainer exec \
    --bind "$STATS_PATCH:/usr/local/lib/python3.10/dist-packages/mlagents/trainers/stats.py" \
    --bind "$TRAINER_PATCH:/usr/local/lib/python3.10/dist-packages/mlagents/trainers/trainer_controller.py" \
    "${CPO_BINDS[@]}" \
    "$SIF" env "${CPO_ENV[@]}" xvfb-run -a mlagents-learn "$CONFIG" --env="$BUILD" --run-id="$RUN_ID" \
        --results-dir="$ROOT/results" --no-graphics --torch-device=cpu --num-envs="$NUM_ENVS" \
        --base-port="$PORT_BASE" --seed="$SEED" --env-args -job-worker-count 1 -traffic-car-count "$TRAIN_CARS"
test -s "$RUN_DIR/CarAgent.onnx"
echo "Done: $RUN_ID"
