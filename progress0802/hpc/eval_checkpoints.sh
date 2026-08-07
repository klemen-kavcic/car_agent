#!/bin/bash
#SBATCH --job-name=car_eval
#SBATCH --partition=all
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --array=0-89
#SBATCH --output=/d/hpc/home/kk42117/mag/car/logs/eval_%A_%a.out
#SBATCH --error=/d/hpc/home/kk42117/mag/car/logs/eval_%A_%a.err

mkdir -p /d/hpc/home/kk42117/mag/car/logs

SIF=/d/hpc/home/kk42117/mag/car/mlagents.sif
BINARY=/d/hpc/home/kk42117/mag/car/Linux/car.x86_64
RESULTS=/d/hpc/home/kk42117/mag/car/results
EVAL_SCRIPT=/d/hpc/home/kk42117/mag/car/eval_checkpoints.py

# One array task per results/<run_id> directory, sorted for a stable index -> run_id mapping.
mapfile -t RUN_DIRS < <(find "$RESULTS" -mindepth 1 -maxdepth 1 -type d | sort)
RUN_DIR="${RUN_DIRS[$SLURM_ARRAY_TASK_ID]}"

if [ -z "$RUN_DIR" ]; then
    echo "No run dir for array index $SLURM_ARRAY_TASK_ID (found ${#RUN_DIRS[@]} run dirs). Adjust --array range."
    exit 1
fi

BASE_PORT=$((7005 + SLURM_ARRAY_TASK_ID * 200))

echo "Evaluating: $RUN_DIR"
echo "  base_port=$BASE_PORT"

apptainer exec "$SIF" \
    xvfb-run -a python "$EVAL_SCRIPT" \
        --binary "$BINARY" \
        --run-dir "$RUN_DIR" \
        --behavior-name CarAgent \
        --episodes 50 \
        --time-scale 20 \
        --base-port "$BASE_PORT" \
        --seed 0 \
        --out "$RUN_DIR/eval_results.csv"
