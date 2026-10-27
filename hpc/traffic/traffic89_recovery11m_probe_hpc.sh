#!/bin/bash
# Five-seed diagnostic run for the 11M recovery curriculum only.
# S-curve and turn each receive 0.4M steps; all other lesson durations match
# the 10.6M curriculum. It ends immediately after close-range deadlock.
# Submit: sbatch traffic89_recovery11m_probe_hpc.sh
#SBATCH --job-name=traffic89_rec11m
#SBATCH --partition=all
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=17:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --array=0-4%6
#SBATCH --output=/d/hpc/home/kk42117/mag/car1/logs/traffic89_rec11m_%A_%a.out
#SBATCH --error=/d/hpc/home/kk42117/mag/car1/logs/traffic89_rec11m_%A_%a.err
set -euo pipefail

ROOT=/d/hpc/home/kk42117/mag/car1
export RECOVERY_PROBE_STEPS=11000000
export RECOVERY_PROBE_LABEL=recovery11m
exec bash "$ROOT/traffic89_recovery75m_probe_hpc.sh"
