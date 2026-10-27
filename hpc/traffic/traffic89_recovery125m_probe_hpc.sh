#!/bin/bash
# Five-seed diagnostic run for the 12.5M recovery curriculum only.
# It includes the final 1M compact close-range circulation lesson.
# Submit: sbatch traffic89_recovery125m_probe_hpc.sh
#SBATCH --job-name=traffic89_rec125m
#SBATCH --partition=all
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=18:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --array=0-4%6
#SBATCH --output=/d/hpc/home/kk42117/mag/car1/logs/traffic89_rec125m_%A_%a.out
#SBATCH --error=/d/hpc/home/kk42117/mag/car1/logs/traffic89_rec125m_%A_%a.err
set -euo pipefail

ROOT=/d/hpc/home/kk42117/mag/car1
export RECOVERY_PROBE_STEPS=12500000
export RECOVERY_PROBE_LABEL=recovery125m
exec bash "$ROOT/traffic89_recovery75m_probe_hpc.sh"
