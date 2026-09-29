#!/bin/bash
# Complete the unfinished 0918 traffic work in one restart-safe Slurm array.
#
#  0..29: final evaluation + trajectories for completed 77 runs
# 30..44: final evaluation + trajectories for 89 runs; retrain only missing ones
# 45..49: fill missing/partial 1M evaluations of the five old 3 m/s models
#
# Valid 250-episode CSVs are skipped, so this script can safely be resubmitted.
#SBATCH --job-name=traffic_resume
#SBATCH --partition=all
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --array=0-49
#SBATCH --output=/d/hpc/home/kk42117/mag/car1/logs/traffic_resume_%A_%a.out
#SBATCH --error=/d/hpc/home/kk42117/mag/car1/logs/traffic_resume_%A_%a.err
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1

ROOT=/d/hpc/home/kk42117/mag/car1
SIF="$ROOT/mlagents.sif"
TRAIN77_ID=${TRAIN77_ID:-19415953}
TRAIN89_ID=${TRAIN89_ID:-19416041}
OLD_CURVE_ID=${OLD_CURVE_ID:-19416438}
TASK=${SLURM_ARRAY_TASK_ID:?array task required}
EPISODES=${TRAFFIC_EVAL_EPISODES:-250}
[[ "$TRAIN77_ID" =~ ^[0-9]+$ && "$TRAIN89_ID" =~ ^[0-9]+$ && "$OLD_CURVE_ID" =~ ^[0-9]+$ && "$EPISODES" =~ ^[1-9][0-9]*$ ]] || { echo 'ERROR: invalid job ID or episode count' >&2; exit 1; }

EVALUATOR="$ROOT/eval_traffic.py"
SELECTOR="$ROOT/select_multiagent_checkpoint.py"
PLOTTER="$ROOT/plot_multiagent_trajectories.py"
PREPARE="$ROOT/prepare_multiagent_config.py"
STATS_PATCH="$ROOT/stats_patched.py"
TRAINER_PATCH="$ROOT/trainer_controller_patched.py"
MANIFEST="$ROOT/traffic_validation_maps.json"
MAPS_DIR=${TRAFFIC_MAPS_DIR:-$ROOT/VoronoiVal}
PORT_SERIAL=0

for f in "$SIF" "$EVALUATOR" "$SELECTOR" "$PLOTTER" "$PREPARE" "$MANIFEST" "$STATS_PATCH" "$TRAINER_PATCH"; do
    [[ -s "$f" ]] || { echo "ERROR: missing or empty $f" >&2; exit 1; }
done

csv_complete() {
    [[ -s "$1" ]] || return 1
    local lines
    lines=$(wc -l < "$1")
    (( lines == EPISODES + 1 ))
}

# Return BASE|LOCK for a free block. A new block is used for every Unity launch:
# Unity can linger briefly after Python exits, which caused the 0918 collisions.
acquire_port_block() {
    # This function runs through command substitution, hence in a subshell.
    # Keep the launch serial in the parent and pass it in explicitly.
    local serial="$1" attempt slot lock base offset free preferred
    preferred=$(((SLURM_ARRAY_JOB_ID * 31 + TASK * 97 + serial * 13) % 800))
    for ((attempt=0; attempt<800; attempt++)); do
        slot=$(((preferred + attempt) % 800))
        lock="/tmp/car_mlagents_portblock_${slot}.lock"
        if mkdir "$lock" 2>/dev/null; then
            base=$((10000 + slot * 64)); free=1
            if command -v ss >/dev/null 2>&1; then
                for ((offset=0; offset<64; offset++)); do
                    if ss -H -ltn "sport = :$((base + offset))" | grep -q .; then free=0; break; fi
                done
            fi
            if ((free)); then printf '%s|%s\n' "$base" "$lock"; return 0; fi
            rmdir "$lock" 2>/dev/null || true
        fi
    done
    return 1
}

run_evaluator() {
    local info base lock status
    info=$(acquire_port_block "$PORT_SERIAL") || { echo 'ERROR: no free ML-Agents port block' >&2; return 1; }
    PORT_SERIAL=$((PORT_SERIAL + 1))
    base=${info%%|*}; lock=${info#*|}
    set +e
    apptainer exec "$SIF" xvfb-run -a env \
        PYTHONPATH="/d/hpc/home/kk42117/.local/lib/python3.10/site-packages:${PYTHONPATH:-}" \
        python3 "$EVALUATOR" "$@" --base-port "$base" --worker-id-base 0 \
        --unity-job-worker-count 1 --time-scale 20 --no-graphics
    status=$?
    set -e
    rmdir "$lock" 2>/dev/null || true
    return "$status"
}

check_build() {
    local build="$1" info="$2" observation="$3"
    [[ -s "$build" && -s "$info" ]] || { echo "ERROR: missing build $build" >&2; exit 1; }
    grep -Fq 'Traffic measurement schema: multi-car-random-placement-v4' "$info" || { echo 'ERROR: wrong validation build schema' >&2; exit 1; }
    grep -Fq 'Traffic car pool size: 10;' "$info" || { echo 'ERROR: validation build needs ten-car pool' >&2; exit 1; }
    grep -Fq "Configured Vector Observation Size: ${observation}  [MATCH]" "$info" || { echo "ERROR: validation build is not ${observation} observations" >&2; exit 1; }
}

run_final_evaluation() {
    local model="$1" config="$2" build="$3" observation="$4" out_root="$5" label="$6"
    for test_cars in 2 5 10; do
        for mode in deterministic stochastic; do
            local out="$out_root/cars${test_cars}/${mode}"
            mkdir -p "$out"
            if ! csv_complete "$out/traffic.csv"; then
                echo "Evaluating $label: $test_cars cars, $mode"
                run_evaluator --build "$build" --manifest "$MANIFEST" --models "$model" --config "$config" \
                    --num-cars "$test_cars" --mode "$mode" --episodes "$EPISODES" \
                    --expected-observations "$observation" --require-tile-metrics --require-map-metrics \
                    --require-random-placement --out "$out/traffic.csv"
            fi
            local traj="$out/trajectories" rollouts=1
            [[ "$mode" == stochastic ]] && rollouts=5
            if [[ ! -s "$traj/trajectories.csv" ]]; then
                mkdir -p "$traj"
                echo "Trajectories $label: $test_cars cars, $mode"
                run_evaluator --build "$build" --manifest "$MANIFEST" --models "$model" --config "$config" \
                    --num-cars "$test_cars" --mode "$mode" --scenario-indices 0 5 6 37 41 43 50 79 83 93 \
                    --rollouts-per-map "$rollouts" --fixed-placement-across-rollouts \
                    --trajectories-out "$traj/trajectories.csv" --trajectory-stride-steps 5 \
                    --expected-observations "$observation" --require-tile-metrics --require-map-metrics \
                    --require-random-placement --out "$traj/traffic.csv"
            fi
        done
        apptainer exec "$SIF" python3 "$PLOTTER" "$out_root/cars${test_cars}" \
            --maps-dir "$MAPS_DIR" --out "$out_root/cars${test_cars}/trajectory_grid.png"
    done
}

if ((TASK < 30)); then
    # Completed 77 run: only the final evaluation/trajectories are missing.
    DENSITIES=(2 5 10); group=$((TASK / 10)); within=$((TASK % 10)); train_cars=${DENSITIES[$group]}
    if ((within < 5)); then method=finetune; seed=$within; else method=scratch; seed=$((within - 5)); fi
    run_id="traffic${train_cars}_${TRAIN77_ID}_${method}_s${seed}"
    run_dir="$ROOT/results/$run_id"
    model="$run_dir/CarAgent.onnx"; config="$run_dir/configuration.yaml"
    check_build "$ROOT/Linux_traffic_all_validation/car.x86_64" "$ROOT/Linux_traffic_all_validation/build_info.txt" 77
    [[ -s "$model" && -s "$config" ]] || { echo "ERROR: 77 model/config missing for $run_id" >&2; exit 1; }
    run_final_evaluation "$model" "$config" "$ROOT/Linux_traffic_all_validation/car.x86_64" 77 \
        "$ROOT/traffic_results/multiagent_final_${TRAIN77_ID}/$run_id" "$run_id"

elif ((TASK < 45)); then
    # Evaluate completed 89 models; fully retrain only 0,2,7,13 (or any model
    # that is absent) in a new run directory so no partial result is overwritten.
    subtask=$((TASK - 30)); DENSITIES=(2 5 10); group=$((subtask / 5)); seed=$((subtask % 5)); train_cars=${DENSITIES[$group]}
    original_id="traffic89_${TRAIN89_ID}_scratch_cars${train_cars}_s${seed}"
    run_id="$original_id"; run_dir="$ROOT/results/$run_id"
    model="$run_dir/CarAgent.onnx"; config="$run_dir/configuration.yaml"
    training_build="$ROOT/Linux_traffic_training_vehicle/car.x86_64"; training_info="$ROOT/Linux_traffic_training_vehicle/build_info.txt"
    validation_build="$ROOT/Linux_traffic_all_validation_vehicle/car.x86_64"; validation_info="$ROOT/Linux_traffic_all_validation_vehicle/build_info.txt"
    check_build "$validation_build" "$validation_info" 89
    if [[ ! -s "$model" ]]; then
        run_id="traffic89_${TRAIN89_ID}_rerun_scratch_cars${train_cars}_s${seed}"
        run_dir="$ROOT/results/$run_id"; model="$run_dir/CarAgent.onnx"; config="$ROOT/configs/${run_id}.yaml"
        [[ -s "$training_build" && -s "$training_info" ]] || { echo 'ERROR: missing 89 training build' >&2; exit 1; }
        grep -Fq 'Traffic car pool size: 10;' "$training_info" || { echo 'ERROR: 89 training build needs ten-car pool' >&2; exit 1; }
        grep -Fq 'Configured Vector Observation Size: 89  [MATCH]' "$training_info" || { echo 'ERROR: wrong 89 training build' >&2; exit 1; }
        source_config="$ROOT/results/car_vgrid_19016936_laser12_cap3_s${seed}/configuration.yaml"
        [[ -s "$source_config" ]] || { echo "ERROR: missing source config $source_config" >&2; exit 1; }
        mkdir -p "$ROOT/configs"
        if ((train_cars == 2)); then train_envs=32
        elif ((train_cars == 5)); then train_envs=12
        else train_envs=6
        fi
        apptainer exec "$SIF" python3 "$PREPARE" --source "$source_config" --out "$config" --steps 40000000 \
            --num-envs "$train_envs" --vehicle-distance
        info=$(acquire_port_block "$PORT_SERIAL") || { echo 'ERROR: no free training port block' >&2; exit 1; }
        PORT_SERIAL=$((PORT_SERIAL + 1))
        base=${info%%|*}; lock=${info#*|}
        set +e
        train_restart_flag=()
        if [[ -s "$run_dir/CarAgent/checkpoint.pt" ]]; then
            # A genuine partial training run exists: continue it.
            train_restart_flag=(--resume)
        elif [[ -e "$run_dir" ]]; then
            # The previous attempt failed before writing a checkpoint.  This
            # rerun-only directory has no usable training state, so ML-Agents
            # must replace its bookkeeping files before it can start fresh.
            train_restart_flag=(--force)
        fi
        apptainer exec \
            --bind "$STATS_PATCH:/usr/local/lib/python3.10/dist-packages/mlagents/trainers/stats.py" \
            --bind "$TRAINER_PATCH:/usr/local/lib/python3.10/dist-packages/mlagents/trainers/trainer_controller.py" \
            "$SIF" xvfb-run -a mlagents-learn "$config" "${train_restart_flag[@]}" --env="$training_build" --run-id="$run_id" \
                --results-dir="$ROOT/results" --no-graphics --torch-device=cpu \
                --num-envs="$train_envs" \
                --base-port="$base" --seed="$seed" --env-args -job-worker-count 1 -traffic-car-count "$train_cars"
        status=$?
        set -e
        rmdir "$lock" 2>/dev/null || true
        ((status == 0)) || exit "$status"
        test -s "$model"
        python3 "$SELECTOR" --run-dir "$run_dir" --max-steps 40000000 --validate-all
    fi
    [[ -s "$config" ]] || { echo "ERROR: missing 89 configuration for $run_id" >&2; exit 1; }
    run_final_evaluation "$model" "$config" "$validation_build" 89 \
        "$ROOT/traffic_results/vehicle89_${TRAIN89_ID}/$run_id" "$run_id"

else
    # Historical 3 m/s single-agent checkpoints: overwrite partial files and
    # evaluate only the missing 250-episode conditions.
    seed=$((TASK - 45)); run_id="car_vgrid_19016936_laser12_cap3_s${seed}"
    run_dir="$ROOT/results/$run_id"; config="$run_dir/configuration.yaml"
    check_build "$ROOT/Linux_traffic_all_validation/car.x86_64" "$ROOT/Linux_traffic_all_validation/build_info.txt" 77
    [[ -s "$config" ]] || { echo "ERROR: missing configuration for $run_id" >&2; exit 1; }
    python3 "$SELECTOR" --run-dir "$run_dir" --max-steps 20000000 --validate-all
    out_root="$ROOT/traffic_results/original_laser12_cap3_checkpoint_curves_${OLD_CURVE_ID}/$run_id"
    for ((milestone=1; milestone<=20; milestone++)); do
        model=$(python3 "$SELECTOR" --run-dir "$run_dir" --max-steps 20000000 --milestone "$milestone")
        for test_cars in 2 5 10; do
            for mode in deterministic stochastic; do
                out="$out_root/step_${milestone}/cars${test_cars}/${mode}"
                if ! csv_complete "$out/traffic.csv"; then
                    mkdir -p "$out"
                    echo "Historical curve $run_id: ${milestone}M, $test_cars cars, $mode"
                    run_evaluator --build "$ROOT/Linux_traffic_all_validation/car.x86_64" --manifest "$MANIFEST" \
                        --models "$model" --config "$config" --num-cars "$test_cars" --mode "$mode" --episodes "$EPISODES" \
                        --expected-observations 77 --require-tile-metrics --require-map-metrics \
                        --require-random-placement --out "$out/traffic.csv"
                fi
            done
        done
    done
fi

echo "Resume task $TASK complete."
