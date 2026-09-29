#!/bin/bash
# Train the five shared 89-observation solo base policies for 20M steps.
# Final traffic evaluation is submitted as a dependent job by the pipeline.
#SBATCH --job-name=traffic89_solo
#SBATCH --partition=all
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --array=0-4%3
#SBATCH --output=/d/hpc/home/kk42117/mag/car1/logs/traffic89_solo_%A_%a.out
#SBATCH --error=/d/hpc/home/kk42117/mag/car1/logs/traffic89_solo_%A_%a.err
set -euo pipefail
ROOT=/d/hpc/home/kk42117/mag/car1
SIF=$ROOT/mlagents.sif
BUILD=$ROOT/Linux_traffic_training_vehicle/car.x86_64
INFO=$ROOT/Linux_traffic_training_vehicle/build_info.txt
PREPARE=$ROOT/prepare_multiagent_config.py
SELECTOR=$ROOT/select_multiagent_checkpoint.py
STATS=$ROOT/stats_patched.py
TRAINER=$ROOT/trainer_controller_patched.py
SEED=$SLURM_ARRAY_TASK_ID
RUN_ID=traffic89_solo20m_s$SEED
RUN_DIR=$ROOT/results/$RUN_ID
CONFIG=$ROOT/configs/$RUN_ID.yaml
SOURCE=$ROOT/results/car_vgrid_19016936_laser12_cap3_s$SEED/configuration.yaml
for f in $SIF $BUILD $INFO $PREPARE $SELECTOR $STATS $TRAINER $SOURCE; do
    test -s $f || { echo "Missing $f" >&2; exit 1; }
done
grep -Fq 'Configured Vector Observation Size: 89  [MATCH]' $INFO || exit 1
mkdir -p $ROOT/configs $ROOT/results
python3 $PREPARE --source $SOURCE --out $CONFIG --steps 20000000 --num-envs 64 --vehicle-distance

# A port can become occupied between the ss check and ML-Agents binding all 64
# workers.  Retry the *whole startup* on a different checked block.  This is
# particularly important for arrays sharing a compute node.  --force is used
# only after a failed startup which created an incomplete run directory; a
# valid checkpoint is never overwritten, unless the caller explicitly sets
# FORCE_RESTART=1 for a known-bad run (for example, a run made before a scene
# bug was fixed).
reserve_port_block() {
    PORT_LOCK=
    local preferred=$1
    for ((SEARCH=0; SEARCH<800; SEARCH++)); do
        local slot=$(((preferred + SEARCH) % 800))
        local lock=/tmp/car_mlagents_portblock_${slot}.lock
        if mkdir "$lock" 2>/dev/null; then
            local candidate_base=$((10000 + slot * 64)) free=1
            if command -v ss >/dev/null 2>&1; then
                for ((OFFSET=0; OFFSET<64; OFFSET++)); do
                    if ss -H -ltn "sport = :$((candidate_base + OFFSET))" | grep -q .; then
                        free=0; break
                    fi
                done
            fi
            if ((free)); then
                PORT_LOCK=$lock; PORT_BASE=$candidate_base; return 0
            fi
            rmdir "$lock" 2>/dev/null || true
        fi
    done
    return 1
}

if [[ -s $RUN_DIR/CarAgent/checkpoint.pt && ${FORCE_RESTART:-0} != 1 ]]; then
    echo "Existing checkpoint found for $RUN_ID; refusing to overwrite it." >&2
    echo "Set FORCE_RESTART=1 only when this exact run is known to be invalid." >&2
    exit 1
fi
if [[ ${FORCE_RESTART:-0} == 1 && -d $RUN_DIR ]]; then
    echo "FORCE_RESTART=1: replacing the known-invalid previous contents of $RUN_ID."
fi

TRAIN_STATUS=1
INITIAL_SLOT=$(((SLURM_ARRAY_JOB_ID * 31 + SEED) % 800))
for ((LAUNCH_ATTEMPT=0; LAUNCH_ATTEMPT<4; LAUNCH_ATTEMPT++)); do
    PREFERRED_SLOT=$(((INITIAL_SLOT + LAUNCH_ATTEMPT * 97) % 800))
    reserve_port_block "$PREFERRED_SLOT" || { echo 'ERROR: no free ML-Agents port block' >&2; exit 1; }
    echo "Launch attempt $((LAUNCH_ATTEMPT + 1))/4 using base port $PORT_BASE"
    FORCE=()
    [[ -d $RUN_DIR ]] && FORCE=(--force)
    set +e
    apptainer exec --bind $STATS:/usr/local/lib/python3.10/dist-packages/mlagents/trainers/stats.py \
        --bind $TRAINER:/usr/local/lib/python3.10/dist-packages/mlagents/trainers/trainer_controller.py \
        $SIF xvfb-run -a mlagents-learn $CONFIG --env=$BUILD --run-id=$RUN_ID \
        --results-dir=$ROOT/results --no-graphics --torch-device=cpu --num-envs=64 \
        --base-port=$PORT_BASE --seed=$SEED "${FORCE[@]}" \
        --env-args -job-worker-count 1 -traffic-car-count 1
    TRAIN_STATUS=$?
    set -e
    rmdir "$PORT_LOCK" 2>/dev/null || true
    PORT_LOCK=
    # Use an explicit conditional: with `set -e`, a failed arithmetic command
    # followed by `&& break` can terminate the script before the retry path.
    if (( TRAIN_STATUS == 0 )); then
        break
    fi
    echo "ML-Agents startup/training exited $TRAIN_STATUS; retrying with a new port block." >&2
    sleep 2
done
(( TRAIN_STATUS == 0 )) || exit "$TRAIN_STATUS"
test -s $RUN_DIR/CarAgent.onnx
python3 $SELECTOR --run-dir $RUN_DIR --max-steps 20000000 --validate-all
