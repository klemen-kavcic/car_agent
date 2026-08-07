#!/bin/bash
#SBATCH --job-name=car_grid
#SBATCH --partition=all
#SBATCH --cpus-per-task=128
#SBATCH --mem=128G
#SBATCH --time=15:00:00
#SBATCH --array=0-89
#SBATCH --output=/d/hpc/home/kk42117/mag/car/logs/train_%A_%a.out
#SBATCH --error=/d/hpc/home/kk42117/mag/car/logs/train_%A_%a.err

mkdir -p /d/hpc/home/kk42117/mag/car/logs /d/hpc/home/kk42117/mag/car/configs

SIF=/d/hpc/home/kk42117/mag/car/mlagents.sif
BINARY=/d/hpc/home/kk42117/mag/car/Linux/car.x86_64
RESULTS=/d/hpc/home/kk42117/mag/car/results

COMBO_IDX=$((SLURM_ARRAY_TASK_ID / 5))
SEED=$((SLURM_ARRAY_TASK_ID % 5))

# 18 combinations = 9 penalty/goal pairs × 2 idle penalty states
# COMBO_IDX 0-8: idle=off, COMBO_IDX 9-17: idle=on
IDLE_IDX=$((COMBO_IDX / 9))
PG_IDX=$((COMBO_IDX % 9))

# Combination order — first 4 are diagonal + (penalty_esc, goal_low):
#   0: low/low    1: high/high   2: esc/esc    3: esc/low
#   4: low/high   5: low/esc     6: high/low   7: high/esc   8: esc/high
case $PG_IDX in
    0) PENALTY_MODE=low;  GOAL_MODE=low  ;;
    1) PENALTY_MODE=high; GOAL_MODE=high ;;
    2) PENALTY_MODE=esc;  GOAL_MODE=esc  ;;
    3) PENALTY_MODE=esc;  GOAL_MODE=low  ;;
    4) PENALTY_MODE=high; GOAL_MODE=low  ;;
    5) PENALTY_MODE=low;  GOAL_MODE=esc  ;;
    6) PENALTY_MODE=low;  GOAL_MODE=high ;;
    7) PENALTY_MODE=high; GOAL_MODE=esc  ;;
    8) PENALTY_MODE=esc;  GOAL_MODE=high ;;
esac

# Penalty: low = -5 fixed, high = -15 fixed, esc = starts -5 and escalates to -15
case $PENALTY_MODE in
    low)  PENALTY_VAL=-5.0;  PENALTY_ESC=0 ;;
    high) PENALTY_VAL=-15.0; PENALTY_ESC=0 ;;
    esc)  PENALTY_VAL=-5.0;  PENALTY_ESC=1 ;;
esac

# Goal reward: low = 7 fixed, high = 16 fixed, esc = starts 7 and escalates to 16
case $GOAL_MODE in
    low)  GOAL_VAL=7.0;  GOAL_ESC=0 ;;
    high) GOAL_VAL=16.0; GOAL_ESC=0 ;;
    esc)  GOAL_VAL=7.0;  GOAL_ESC=1 ;;
esac

# Idle penalty: off = disabled (0), on = enabled (1)
case $IDLE_IDX in
    0) IDLE_VAL=0 ;;
    1) IDLE_VAL=1 ;;
esac
IDLE_MODE=$([[ $IDLE_IDX -eq 0 ]] && echo "off" || echo "on")

RUN_ID="car_${SLURM_ARRAY_JOB_ID}_p${PENALTY_MODE}_g${GOAL_MODE}_i${IDLE_MODE}_s${SEED}"
BASE_PORT=$((5005 + SLURM_ARRAY_TASK_ID * 200))

CONFIG=/d/hpc/home/kk42117/mag/car/configs/car_config_${SLURM_JOB_ID}.yaml
sed \
    -e "s/reward_goal: .*/reward_goal: $GOAL_VAL/" \
    -e "s/reward_terminal_penalty: .*/reward_terminal_penalty: $PENALTY_VAL/" \
    -e "s/terminal_penalty_escalation_enabled: .*/terminal_penalty_escalation_enabled: $PENALTY_ESC/" \
    -e "s/goal_reward_escalation_enabled: .*/goal_reward_escalation_enabled: $GOAL_ESC/" \
    -e "s/idle_penalty_enabled: .*/idle_penalty_enabled: $IDLE_VAL/" \
    /d/hpc/home/kk42117/mag/car/car_config.yaml > "$CONFIG"

echo "Run: $RUN_ID"
echo "  penalty=$PENALTY_MODE (val=$PENALTY_VAL, esc=$PENALTY_ESC)"
echo "  goal=$GOAL_MODE    (val=$GOAL_VAL, esc=$GOAL_ESC)"
echo "  idle=$IDLE_MODE   (enabled=$IDLE_VAL)"
echo "  seed=$SEED | port=$BASE_PORT | combo=$COMBO_IDX"

echo "=== Build info ==="
cat /d/hpc/home/kk42117/mag/car/Linux/build_info.txt
echo "=================="

apptainer exec \
    --bind /d/hpc/home/kk42117/mag/car/stats_patched.py:/usr/local/lib/python3.10/dist-packages/mlagents/trainers/stats.py \
    "$SIF" \
    xvfb-run -a mlagents-learn "$CONFIG" \
        --env="$BINARY" \
        --run-id="$RUN_ID" \
        --results-dir="$RESULTS" \
        --no-graphics \
        --torch-device=cpu \
        --num-envs=32 \
        --base-port=$BASE_PORT \
        --seed=$SEED
