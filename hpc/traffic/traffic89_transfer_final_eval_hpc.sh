#!/bin/bash
# Final 250-episode evaluation plus trajectories for the shared-solo transfer
# experiment. Tasks 0..4 evaluate solo bases; tasks 5..19 evaluate fine-tunes.
# The submit wrapper releases the respective task ranges after training.
#SBATCH --job-name=traffic89_transfer_eval
#SBATCH --partition=all
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --array=0-19%3
#SBATCH --output=/d/hpc/home/kk42117/mag/car1/logs/traffic89_transfer_eval_%A_%a.out
#SBATCH --error=/d/hpc/home/kk42117/mag/car1/logs/traffic89_transfer_eval_%A_%a.err
set -euo pipefail
ROOT=/d/hpc/home/kk42117/mag/car1
SIF=$ROOT/mlagents.sif
POST=$ROOT/traffic_vehicle_distance_post_train_eval.sh
TASK=$SLURM_ARRAY_TASK_ID
if ((TASK < 5)); then
    SEED=$TASK
    RUN_ID=traffic89_solo20m_s$SEED
else
    SUB=$((TASK - 5)); GROUP=$((SUB / 5)); SEED=$((SUB % 5))
    if ((GROUP == 0)); then TRAIN_CARS=2
    elif ((GROUP == 1)); then TRAIN_CARS=5
    else TRAIN_CARS=10
    fi
    RUN_ID=traffic89_solo20m_then_traffic20m_cars$TRAIN_CARS-s$SEED
fi
RUN_DIR=$ROOT/results/$RUN_ID
CONFIG=$ROOT/configs/$RUN_ID.yaml
for f in $SIF $POST $RUN_DIR/CarAgent.onnx $CONFIG; do
    test -s $f || { echo "Missing $f" >&2; exit 1; }
done
PORT_LOCK=
PREFERRED_SLOT=$(((SLURM_ARRAY_JOB_ID * 31 + TASK) % 800))
for ((ATTEMPT=0; ATTEMPT<800; ATTEMPT++)); do
    SLOT=$(((PREFERRED_SLOT + ATTEMPT) % 800))
    LOCK=/tmp/car_mlagents_portblock_$SLOT.lock
    if mkdir $LOCK 2>/dev/null; then
        CANDIDATE_BASE=$((10000 + SLOT * 64)); FREE=1
        if command -v ss >/dev/null 2>&1; then
            for ((OFFSET=0; OFFSET<64; OFFSET++)); do
                if ss -H -ltn "sport = :$((CANDIDATE_BASE + OFFSET))" | grep -q .; then FREE=0; break; fi
            done
        fi
        if ((FREE)); then PORT_LOCK=$LOCK; PORT_BASE=$CANDIDATE_BASE; break; fi
        rmdir $LOCK 2>/dev/null || true
    fi
done
[[ -n $PORT_LOCK ]] || { echo 'ERROR: no free ML-Agents port block' >&2; exit 1; }
cleanup(){ rmdir $PORT_LOCK 2>/dev/null || true; }
trap cleanup EXIT

# Run the six independent 250-episode final evaluations simultaneously.  The
# trajectory helper below sees these complete CSVs and only collects the
# smaller fixed-map trajectories sequentially.
BUILD=$ROOT/Linux_traffic_all_validation_vehicle/car.x86_64
MANIFEST=$ROOT/traffic_validation_maps.json
EVALUATOR=$ROOT/eval_traffic.py
for f in $BUILD $MANIFEST $EVALUATOR; do
    test -s $f || { echo "Missing $f" >&2; exit 1; }
done
reserve_parallel_port() {
    local serial=$1 attempt slot lock base offset free preferred
    preferred=$(((SLURM_ARRAY_JOB_ID * 97 + TASK * 31 + serial * 13) % 800))
    for ((attempt=0; attempt<800; attempt++)); do
        slot=$(((preferred + attempt) % 800))
        lock=/tmp/car_mlagents_portblock_$slot.lock
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
PIDS=""; PARLOCKS=""; INDEX=0
for CARS in 2 5 10; do
    for MODE in deterministic stochastic; do
        OUT=$ROOT/traffic_results/vehicle89_$SLURM_ARRAY_JOB_ID/$RUN_ID/cars$CARS/$MODE/traffic.csv
        if [[ -s $OUT ]] && (( $(wc -l < $OUT) == 251 )); then
            continue
        fi
        INFO=$(reserve_parallel_port $INDEX) || { echo 'ERROR: no evaluation port block' >&2; exit 1; }
        EVAL_BASE=$(printf '%s' "$INFO" | cut -d '|' -f 1)
        EVAL_LOCK=$(printf '%s' "$INFO" | cut -d '|' -f 2)
        PARLOCKS="$PARLOCKS $EVAL_LOCK"
        mkdir -p $(dirname $OUT)
        apptainer exec $SIF xvfb-run -a env \
            PYTHONPATH=/d/hpc/home/kk42117/.local/lib/python3.10/site-packages \
            python3 $EVALUATOR --build $BUILD --manifest $MANIFEST \
            --models $RUN_DIR/CarAgent.onnx --config $CONFIG --num-cars $CARS \
            --mode $MODE --episodes 250 --expected-observations 89 \
            --require-tile-metrics --require-map-metrics --require-random-placement \
            --base-port $EVAL_BASE --worker-id-base 0 --unity-job-worker-count 1 \
            --time-scale 20 --no-graphics --out $OUT &
        PIDS="$PIDS $!"
        INDEX=$((INDEX + 1))
    done
done
FAIL=0
for PID in $PIDS; do wait $PID || FAIL=1; done
for LOCK in $PARLOCKS; do rmdir $LOCK 2>/dev/null || true; done
((FAIL == 0)) || { echo "ERROR: parallel final evaluation failed for $RUN_ID" >&2; exit 1; }
export ROOT SIF PORT_BASE
bash $POST $RUN_ID $RUN_DIR/CarAgent.onnx $CONFIG
