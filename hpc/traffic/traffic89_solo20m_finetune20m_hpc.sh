#!/bin/bash
# 89 observations: 20M single-car pretraining, then 20M multi-agent fine-tuning.
# Five seeds x training densities 2/5/10.
#SBATCH --job-name=traffic89_finetune
#SBATCH --partition=all
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --array=0-14%3
#SBATCH --output=/d/hpc/home/kk42117/mag/car1/logs/traffic89_finetune_%A_%a.out
#SBATCH --error=/d/hpc/home/kk42117/mag/car1/logs/traffic89_finetune_%A_%a.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
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
if ((GROUP == 0)); then TRAIN_CARS=2
elif ((GROUP == 1)); then TRAIN_CARS=5
else TRAIN_CARS=10
fi
RUN_ID=traffic89_ft_cars$TRAIN_CARS-s$SEED
RUN_DIR=$ROOT/results/$RUN_ID
CONFIG=$ROOT/configs/$RUN_ID.yaml
SOURCE=$ROOT/results/car_vgrid_19016936_laser12_cap3_s$SEED/configuration.yaml
for f in $SIF $BUILD $INFO $PREPARE $SELECTOR $STATS $TRAINER $SOURCE; do
    test -s $f || { echo "Missing $f" >&2; exit 1; }
done
grep -Fq 'Configured Vector Observation Size: 89  [MATCH]' $INFO || exit 1
mkdir -p $ROOT/configs $ROOT/results
BASE=$((10000 + TASK * 128))
if ((TRAIN_CARS == 2)); then ENVS=32
elif ((TRAIN_CARS == 5)); then ENVS=12
else ENVS=6
fi
python3 $PREPARE --source $SOURCE --out $CONFIG --steps 20000000 --num-envs 64 --vehicle-distance
echo "Solo phase: $RUN_ID, seed=$SEED, 20M steps"
apptainer exec --bind $STATS:/usr/local/lib/python3.10/dist-packages/mlagents/trainers/stats.py \
    --bind $TRAINER:/usr/local/lib/python3.10/dist-packages/mlagents/trainers/trainer_controller.py \
    $SIF xvfb-run -a mlagents-learn $CONFIG --env=$BUILD --run-id=$RUN_ID \
    --results-dir=$ROOT/results --no-graphics --torch-device=cpu --num-envs=64 \
    --base-port=$BASE --seed=$SEED --env-args -job-worker-count 1 -traffic-car-count 1
python3 $PREPARE --source $SOURCE --out $CONFIG --steps 40000000 --num-envs $ENVS --vehicle-distance
echo "Traffic fine-tune phase: $RUN_ID, cars=$TRAIN_CARS, 20M additional steps"
apptainer exec --bind $STATS:/usr/local/lib/python3.10/dist-packages/mlagents/trainers/stats.py \
    --bind $TRAINER:/usr/local/lib/python3.10/dist-packages/mlagents/trainers/trainer_controller.py \
    $SIF xvfb-run -a mlagents-learn $CONFIG --resume --env=$BUILD --run-id=$RUN_ID \
    --results-dir=$ROOT/results --no-graphics --torch-device=cpu --num-envs=$ENVS \
    --base-port=$BASE --seed=$SEED --env-args -job-worker-count 1 -traffic-car-count $TRAIN_CARS
test -s $RUN_DIR/CarAgent.onnx
python3 $SELECTOR --run-dir $RUN_DIR --max-steps 40000000 --validate-all
echo "Complete: $RUN_ID"
