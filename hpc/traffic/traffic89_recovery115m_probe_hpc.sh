#!/bin/bash
# Five-seed diagnostic run for the 11.5M recovery curriculum only.
# Blind alley and dead-end detour use their restored 1.0M / 0.5M durations.
# Submit: sbatch traffic89_recovery115m_probe_hpc.sh
#SBATCH --job-name=traffic89_rec115m
#SBATCH --partition=all
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=18:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --array=0-4%6
#SBATCH --output=/d/hpc/home/kk42117/mag/car1/logs/traffic89_rec115m_%A_%a.out
#SBATCH --error=/d/hpc/home/kk42117/mag/car1/logs/traffic89_rec115m_%A_%a.err
set -euo pipefail

ROOT=/d/hpc/home/kk42117/mag/car1
export RECOVERY_PROBE_STEPS=11500000
export RECOVERY_PROBE_LABEL=recovery115m
exec bash "$ROOT/traffic89_recovery75m_probe_hpc.sh"
