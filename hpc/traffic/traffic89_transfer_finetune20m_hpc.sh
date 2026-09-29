#!/bin/bash
# Branch each shared 89-observation solo base policy into 2/5/10-car traffic
# fine-tunes. The resulting 15 models train for 20M traffic steps each.
# Final traffic evaluation is submitted as a dependent job by the pipeline.
#SBATCH --job-name=traffic89_transfer
#SBATCH --partition=all
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --array=0-14%3
#SBATCH --output=/d/hpc/home/kk42117/mag/car1/logs/traffic89_transfer_%A_%a.out
#SBATCH --error=/d/hpc/home/kk42117/mag/car1/logs/traffic89_transfer_%A_%a.err
set -euo pipefail
ROOT=/d/hpc/home/kk42117/mag/car1
SIF=$ROOT/mlagents.sif
BUILD=$ROOT/Linux_traffic_training_vehicle/car.x86_64
INFO=$ROOT/Linux_traffic_training_vehicle/build_info.txt
PREPARE=$ROOT/prepare_multiagent_config.py
SELECTOR=$ROOT/select_multiagent_checkpoint.py
STATS=$ROOT/stats_patched.py
TRAINER=$ROOT/trainer_controller_patched.py
TASK=$SLURM_ARRAY_TASK_ID
GROUP=$((TASK / 5)); SEED=$((TASK % 5))
if ((GROUP == 0)); then TRAIN_CARS=2; ENVS=32
elif ((GROUP == 1)); then TRAIN_CARS=5; ENVS=12
else TRAIN_CARS=10; ENVS=6
fi
SOLO_ID=traffic89_solo20m_s$SEED
CHECKPOINT=$ROOT/results/$SOLO_ID/CarAgent/checkpoint.pt
SOURCE=$ROOT/results/car_vgrid_19016936_laser12_cap3_s$SEED/configuration.yaml
RUN_ID=traffic89_solo20m_then_traffic20m_cars$TRAIN_CARS-s$SEED
RUN_DIR=$ROOT/results/$RUN_ID
CONFIG=$ROOT/configs/$RUN_ID.yaml
for f in $SIF $BUILD $INFO $PREPARE $SELECTOR $STATS $TRAINER $SOURCE $CHECKPOINT; do
    test -s $f || { echo "Missing $f" >&2; exit 1; }
done
grep -Fq 'Configured Vector Observation Size: 89  [MATCH]' $INFO || exit 1
mkdir -p $ROOT/configs $ROOT/results
python3 $PREPARE --source $SOURCE --out $CONFIG --steps 20000000 --num-envs $ENVS \
    --checkpoint $CHECKPOINT --vehicle-distance
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
apptainer exec --bind $STATS:/usr/local/lib/python3.10/dist-packages/mlagents/trainers/stats.py \
    --bind $TRAINER:/usr/local/lib/python3.10/dist-packages/mlagents/trainers/trainer_controller.py \
    $SIF xvfb-run -a mlagents-learn $CONFIG --env=$BUILD --run-id=$RUN_ID \
    --results-dir=$ROOT/results --no-graphics --torch-device=cpu --num-envs=$ENVS \
    --base-port=$PORT_BASE --seed=$SEED --env-args -job-worker-count 1 -traffic-car-count $TRAIN_CARS
test -s $RUN_DIR/CarAgent.onnx
python3 $SELECTOR --run-dir $RUN_DIR --max-steps 20000000 --validate-all
