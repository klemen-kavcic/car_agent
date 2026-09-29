#!/bin/bash
# 1M-checkpoint traffic curves for all completed 0919 policies.
# Tasks 0..14: 77 fine-tune (20M); 15..29: 77 scratch (40M);
# 30..44: 89 scratch (40M). Six 250-episode conditions run in parallel.
#SBATCH --job-name=traffic_curve0919
#SBATCH --partition=all
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --array=0-44%3
#SBATCH --output=/d/hpc/home/kk42117/mag/car1/logs/traffic_curve0919_%A_%a.out
#SBATCH --error=/d/hpc/home/kk42117/mag/car1/logs/traffic_curve0919_%A_%a.err
set -euo pipefail
ROOT=/d/hpc/home/kk42117/mag/car1
SIF=$ROOT/mlagents.sif
EVALUATOR=$ROOT/eval_traffic.py
SELECTOR=$ROOT/select_multiagent_checkpoint.py
MANIFEST=$ROOT/traffic_validation_maps.json
TASK=$SLURM_ARRAY_TASK_ID
EPISODES=250
if ((TASK < 15)); then
  SENSOR=77; MAX_M=20; GROUP=$((TASK / 5)); SEED=$((TASK % 5))
  if ((GROUP == 0)); then TRAIN_CARS=2; elif ((GROUP == 1)); then TRAIN_CARS=5; else TRAIN_CARS=10; fi
  PREFIX=traffic$TRAIN_CARS; RUN_ID=$PREFIX"_19415953_finetune_s"$SEED
  BUILD=$ROOT/Linux_traffic_all_validation/car.x86_64
elif ((TASK < 30)); then
  SENSOR=77; MAX_M=40; SUB=$((TASK - 15)); GROUP=$((SUB / 5)); SEED=$((SUB % 5))
  if ((GROUP == 0)); then TRAIN_CARS=2; elif ((GROUP == 1)); then TRAIN_CARS=5; else TRAIN_CARS=10; fi
  PREFIX=traffic$TRAIN_CARS; RUN_ID=$PREFIX"_19415953_scratch_s"$SEED
  BUILD=$ROOT/Linux_traffic_all_validation/car.x86_64
else
  SENSOR=89; MAX_M=40; SUB=$((TASK - 30)); GROUP=$((SUB / 5)); SEED=$((SUB % 5))
  if ((GROUP == 0)); then TRAIN_CARS=2; elif ((GROUP == 1)); then TRAIN_CARS=5; else TRAIN_CARS=10; fi
  PREFIX=traffic89_19416041_scratch_cars$TRAIN_CARS
  RUN_ID=$PREFIX"_s"$SEED
  RERUN=traffic89_19416041_rerun_scratch_cars$TRAIN_CARS"_s"$SEED
  [[ -d $ROOT/results/$RERUN ]] && RUN_ID=$RERUN
  BUILD=$ROOT/Linux_traffic_all_validation_vehicle/car.x86_64
fi
RUN_DIR=$ROOT/results/$RUN_ID
CONFIG=$RUN_DIR/configuration.yaml
INFO=$(dirname $BUILD)/build_info.txt
for f in $SIF $EVALUATOR $SELECTOR $MANIFEST $BUILD $INFO $CONFIG; do test -s $f || exit 1; done
grep -Fq "Configured Vector Observation Size: $SENSOR  [MATCH]" $INFO || exit 1
python3 $SELECTOR --run-dir $RUN_DIR --max-steps $((MAX_M * 1000000)) --validate-all
OUT_ROOT=$ROOT/traffic_results/checkpoint_curves_0919/$RUN_ID
reserve_port() {
  local serial=$1 attempt slot lock base offset free preferred
  preferred=$(((SLURM_ARRAY_JOB_ID * 97 + TASK * 31 + serial * 13) % 800))
  for ((attempt=0; attempt<800; attempt++)); do
    slot=$(((preferred + attempt) % 800)); lock=/tmp/car_mlagents_portblock_$slot.lock
    if mkdir $lock 2>/dev/null; then
      base=$((10000 + slot * 64)); free=1
      if command -v ss >/dev/null 2>&1; then
        for ((offset=0; offset<64; offset++)); do
          if ss -H -ltn "sport = :$((base + offset))" | grep -q .; then free=0; break; fi
        done
      fi
      if ((free)); then printf '%s|%s\n' "$base" "$lock"; return 0; fi
      rmdir $lock 2>/dev/null || true
    fi
  done
  return 1
}
for ((MILESTONE=1; MILESTONE<=MAX_M; MILESTONE++)); do
  MODEL=$(python3 $SELECTOR --run-dir $RUN_DIR --max-steps $((MAX_M * 1000000)) --milestone $MILESTONE)
  PIDS=""; LOCKS=""; INDEX=0
  for CARS in 2 5 10; do
    for MODE in deterministic stochastic; do
      OUT=$OUT_ROOT/step_$MILESTONE/cars$CARS/$MODE/traffic.csv
      if [[ -s $OUT ]] && (( $(wc -l < $OUT) == EPISODES + 1 )); then INDEX=$((INDEX + 1)); continue; fi
      PORT_INFO=$(reserve_port $((MILESTONE * 10 + INDEX))) || exit 1
      BASE=$(printf '%s' "$PORT_INFO" | cut -d '|' -f 1)
      LOCK=$(printf '%s' "$PORT_INFO" | cut -d '|' -f 2)
      LOCKS="$LOCKS $LOCK"; mkdir -p $(dirname $OUT)
      apptainer exec $SIF xvfb-run -a env PYTHONPATH=/d/hpc/home/kk42117/.local/lib/python3.10/site-packages \
        python3 $EVALUATOR --build $BUILD --manifest $MANIFEST --models $MODEL --config $CONFIG \
        --num-cars $CARS --mode $MODE --episodes $EPISODES --expected-observations $SENSOR \
        --require-tile-metrics --require-map-metrics --require-random-placement \
        --base-port $BASE --worker-id-base 0 --unity-job-worker-count 1 --time-scale 20 \
        --no-graphics --out $OUT &
      PIDS="$PIDS $!"; INDEX=$((INDEX + 1))
    done
  done
  FAIL=0
  for PID in $PIDS; do wait $PID || FAIL=1; done
  for LOCK in $LOCKS; do rmdir $LOCK 2>/dev/null || true; done
  ((FAIL == 0)) || exit 1
  echo "Completed $RUN_ID at $MILESTONE M"
done
