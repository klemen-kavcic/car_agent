#!/bin/bash
# Missing 0919 checkpoint curves: 89-observation solo (1--20M) and
# 89-observation solo->traffic fine-tune (21--40M), for 2/5/10 test cars.
# Array tasks 0--14 = solo; 15--29 = fine-tune. Five seeds per family.
# The existing 77 solo curve job remains traffic_original_laser12_cap3_curve_hpc.sh.
#SBATCH --job-name=traffic_missing_curve
#SBATCH --partition=all
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --array=0-29%3
#SBATCH --output=/d/hpc/home/kk42117/mag/car1/logs/traffic_missing_curve_%A_%a.out
#SBATCH --error=/d/hpc/home/kk42117/mag/car1/logs/traffic_missing_curve_%A_%a.err
set -euo pipefail
ROOT=/d/hpc/home/kk42117/mag/car1; SIF=$ROOT/mlagents.sif
EVAL=$ROOT/eval_traffic.py; SEL=$ROOT/select_multiagent_checkpoint.py
MAN=$ROOT/traffic_validation_maps.json; TASK=${SLURM_ARRAY_TASK_ID:?}
EP=250; GROUP=$((TASK/5)); SEED=$((TASK%5)); TEST_GROUP=$((GROUP%3))
if (( TASK < 15 )); then
  FAMILY=solo; MAX=20; TRAIN_CARS=1
  RUN_ID=traffic89_solo20m_s$SEED; BUILD=$ROOT/Linux_traffic_all_validation_vehicle/car.x86_64
else
  FAMILY=finetune; MAX=40; TRAIN_CARS=$((TEST_GROUP==0?2:TEST_GROUP==1?5:10))
  RUN_ID=traffic89_solo20m_then_traffic20m_cars${TRAIN_CARS}-s${SEED}
  BUILD=$ROOT/Linux_traffic_all_validation_vehicle/car.x86_64
fi
RUN=$ROOT/results/$RUN_ID; CONFIG=$RUN/configuration.yaml
[[ -s $BUILD && -s $EVAL && -s $SEL && -s $MAN && -s $CONFIG ]] || exit 1
python3 $SEL --run-dir $RUN --max-steps $((MAX*1000000)) --validate-all >/dev/null
reserve_port() {
  local serial=$1 slot lock base off free
  slot=$(((SLURM_ARRAY_JOB_ID*97+TASK*31+serial*13)%800))
  for ((i=0;i<800;i++)); do
    lock=/tmp/car_mlagents_portblock_$slot.lock
    if mkdir "$lock" 2>/dev/null; then
      base=$((10000+slot*64)); free=1
      if command -v ss >/dev/null 2>&1; then
        for ((off=0;off<64;off++)); do
          ss -H -ltn "sport = :$((base+off))" | grep -q . && { free=0; break; }
        done
      fi
      if ((free)); then printf '%s|%s\n' "$base" "$lock"; return 0; fi
      rmdir "$lock" 2>/dev/null || true
    fi
    slot=$(((slot+1)%800))
  done
  return 1
}
trap 'for l in ${LOCKS:-}; do rmdir "$l" 2>/dev/null || true; done' EXIT INT TERM
start=1; (( FAMILY == finetune )) && start=21
for ((m=start;m<=MAX;m++)); do
  model=$(python3 $SEL --run-dir $RUN --max-steps $((MAX*1000000)) --milestone $m)
  PIDS=""; LOCKS=""; serial=0; fail=0
  for cars in 2 5 10; do for mode in deterministic stochastic; do
    out=$ROOT/traffic_results/checkpoint_curves_0919/$RUN_ID/step_$m/cars$cars/$mode/traffic.csv
    [[ -s $out ]] && continue
    info=$(reserve_port $serial) || exit 1; serial=$((serial+1))
    base=${info%%|*}; lock=${info#*|}; LOCKS="$LOCKS $lock"
    mkdir -p "$(dirname "$out")"
    apptainer exec $SIF xvfb-run -a env PYTHONPATH=/d/hpc/home/kk42117/.local/lib/python3.10/site-packages \
      python3 $EVAL --build $BUILD --manifest $MAN --models $model --config $CONFIG \
      --num-cars $cars --mode $mode --episodes $EP --expected-observations 89 \
      --require-tile-metrics --require-map-metrics --require-random-placement \
      --base-port $base --worker-id-base 0 --unity-job-worker-count 1 --time-scale 20 --no-graphics --out $out &
    PIDS="$PIDS $!"
  done; done
  for p in $PIDS; do wait $p || fail=1; done
  for l in $LOCKS; do rmdir "$l" 2>/dev/null || true; done
  ((fail==0)) || exit 1
done
