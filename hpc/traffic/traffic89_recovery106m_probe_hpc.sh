#!/bin/bash
# Five-seed diagnostic run for the revised 10.6M recovery curriculum only.
# This ends immediately after the close-range deadlock lesson; TensorBoard
# summaries every 10k steps are used to judge every lesson before full training.
# Submit: sbatch traffic89_recovery106m_probe_hpc.sh
#SBATCH --job-name=traffic89_rec106m
#SBATCH --partition=all
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=16:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --array=0-4%6
#SBATCH --output=/d/hpc/home/kk42117/mag/car1/logs/traffic89_rec106m_%A_%a.out
#SBATCH --error=/d/hpc/home/kk42117/mag/car1/logs/traffic89_rec106m_%A_%a.err
set -euo pipefail

ROOT=/d/hpc/home/kk42117/mag/car1
export RECOVERY_PROBE_STEPS=10600000
export RECOVERY_PROBE_LABEL=recovery106m
exec bash "$ROOT/traffic89_recovery75m_probe_hpc.sh"
