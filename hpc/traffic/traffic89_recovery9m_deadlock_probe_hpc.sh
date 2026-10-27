#!/bin/bash
# Five-seed diagnostic run for the full 9M recovery curriculum. The final
# 1.5M is the new four-car close-range deadlock lesson.
# Submit: sbatch traffic89_recovery9m_deadlock_probe_hpc.sh
#SBATCH --job-name=traffic89_rec9m
#SBATCH --partition=all
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=14:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --array=0-4%6
#SBATCH --output=/d/hpc/home/kk42117/mag/car1/logs/traffic89_rec9m_%A_%a.out
#SBATCH --error=/d/hpc/home/kk42117/mag/car1/logs/traffic89_rec9m_%A_%a.err
set -euo pipefail

ROOT=/d/hpc/home/kk42117/mag/car1
export RECOVERY_PROBE_STEPS=9000000
export RECOVERY_PROBE_LABEL=recovery9m_deadlock
exec bash "$ROOT/traffic89_recovery75m_probe_hpc.sh"
