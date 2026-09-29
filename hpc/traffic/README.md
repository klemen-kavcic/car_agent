# Two-car laser traffic pilot

This is a separate evaluation scene. `SampleScene.unity` and the old 57-/77-observation builds remain unchanged.

1. Open the Unity project and wait for scripts to compile. Choose **Tools > Traffic > Create two-car laser test scene**. This creates `Assets/Scenes/TrafficTwoCar.unity` without overwriting `SampleScene.unity`.
2. Open the new scene. It has two independent `TrafficCar0/1` behaviors, one shared map, two goals, and a shared 5000-physics-step episode clock. Set `TrafficScenarioManager.scenarioIndex` to inspect any of the ten cases. Use `hpc/traffic/traffic_pairs_preview.png` to review both planned routes.
3. Build the new scene as its own executable. The old `Linux_laser77` build does **not** contain the traffic code or the second car.
4. In a Python environment containing `mlagents-envs`, `onnxruntime`, `numpy` and `PyYAML`, evaluate the existing 3 m/s, 12-laser checkpoint:

   ```bash
   python hpc/traffic/eval_traffic.py --build /path/to/traffic/car.x86_64 --no-graphics --scenario 0
   ```

   Omit `--scenario` to run all ten cases. To compare two trained seeds in the *same* scenario, add `--model1 hpc/0909/results/car_vgrid_19016936_laser12_cap3_s1/CarAgent.onnx`; `--model0` defaults to seed 0. Use `--mode stochastic` for sampled actions. Both models must have the same 77-input/action shape and compatible vehicle-speed/laser-normalization settings.

5. The evaluator writes one row per case to `hpc/traffic/traffic_eval.csv` with each car's outcome and reward. Run same-seed and mixed-seed cases separately for a fair comparison.

The second route on each map is connected asphalt, has separated endpoints and shares interior road cells with the first route. This creates a *possible* encounter, but the policy is not constrained to follow the shortest path, so an actual meeting is not guaranteed. These old checkpoints were trained without moving vehicles; this run is a diagnostic of transfer, not a fair traffic-trained performance score.

## First measured pass on HPC

Build `TrafficTwoCar.unity` for Linux **after** the `TrafficScenarioManager.cs` encounter-metric changes. In Build Settings, enable only `TrafficTwoCar.unity`. Keep both `Behavior Parameters` in **Default** mode: with Python connected, Python supplies their actions, even if an ONNX asset is assigned for visual Editor testing. `Inference Only` would bypass Python and invalidate this evaluator.

Upload the new Linux build directory to `/d/hpc/home/kk42117/mag/car1/Linux_traffic`, `eval_traffic.py` to `car1/eval_traffic.py`, `traffic_eval_hpc.sh` to `car1/traffic_eval_hpc.sh`, and the current root `eval_checkpoints.py` to `car1/eval_checkpoints.py`. The ten-case JSON is embedded in the Unity build through `Resources`; it does not need a separate HPC copy. The script expects the existing seed-0 cap-3 model and `configuration.yaml` under `car1/results/car_vgrid_19016936_laser12_cap3_s0/`.

On the HPC login node:

```bash
cd /d/hpc/home/kk42117/mag/car1
mkdir -p logs
chmod +x Linux_traffic/car.x86_64
sed -i 's/\r$//' traffic_eval_hpc.sh
ls -lh Linux_traffic/car.x86_64 Linux_traffic/build_info.txt eval_traffic.py eval_checkpoints.py results/car_vgrid_19016936_laser12_cap3_s0/CarAgent.onnx results/car_vgrid_19016936_laser12_cap3_s0/configuration.yaml
sbatch --export=ALL,TRAFFIC_SCENARIO=0 /d/hpc/home/kk42117/mag/car1/traffic_eval_hpc.sh
```

With the updated script, the smoke test produces one row for each action mode in `traffic_results/<job-id>/`. Verify the `.out` and `.err` files in `logs/` and confirm that both outcomes are known. A job submitted before the script update (including the original smoke-test job) keeps its submitted script and runs deterministic only. Then submit all ten with the updated script:

```bash
sbatch /d/hpc/home/kk42117/mag/car1/traffic_eval_hpc.sh
```

The full job writes `traffic_seed0_seed0_deterministic.csv` and `traffic_seed0_seed0_stochastic.csv`, plus one `_summary.csv` for each mode, under `traffic_results/<job-id>/`. Each case row contains both cars' outcomes, rewards and finish steps, the shared episode length, minimum car-centre distance, and physics steps closer than 10 m. The summary counts cases with a vehicle collision separately from affected cars (one two-car crash is one case but two vehicle-collision outcomes). `Terminated` means a non-vehicle failure: terminal tile, wall collision, fall or rollover; the current Unity build cannot split those causes further. This is one rollout per case per mode; use repeated Unity seeds and both seat orders for a stronger comparison after it works.

For a later shared-policy training run, give all cars the same BehaviorName. The ten-case pilot manifest remains two-car; the full-validation generator and evaluator below support a configurable car count without changing the 77-value policy observation shape.

## All five seed combinations, with tile-time diagnostics

The updated `TrafficScenarioManager` records each car's finish tile and active time in seconds on Slippery, SpeedLimited, Terminal, Asphalt, and Gravel. Time stops accumulating for a car once it reaches a goal or otherwise finishes; the other car's clock continues. Thus a timeout row shows both the tile where it timed out and whether it spent much of its episode on a slow surface.

Rebuild **only** `TrafficTwoCar.unity` for Linux after these C# changes and replace `car1/Linux_traffic/` with that complete new build. Its `build_info.txt` must contain `Traffic measurement schema: finish-tile-and-per-seat-tile-seconds-v2`. Upload the updated `eval_traffic.py` as `car1/eval_traffic.py` and `traffic_all_seed_pairs_hpc.sh` as `car1/traffic_all_seed_pairs_hpc.sh`. The unchanged `eval_checkpoints.py` must also be present in `car1/`. No YAML edits or retraining are needed.

The array has 50 tasks: five seed choices for seat 0 times five for seat 1 times deterministic/stochastic. It includes self-pairs and both seat orders for mixed pairs. Each task runs the same ten cases once, for 500 case rollouts total; at most six array tasks run concurrently. Results are separated by pair and mode under `car1/traffic_results/pairwise_<array-job-id>/`.

On the HPC login node, after uploading:

```bash
cd /d/hpc/home/kk42117/mag/car1
mkdir -p logs
chmod +x Linux_traffic/car.x86_64
sed -i 's/\r$//' traffic_all_seed_pairs_hpc.sh
grep -F 'Traffic measurement schema: finish-tile-and-per-seat-tile-seconds-v2' Linux_traffic/build_info.txt
for seed in 0 1 2 3 4; do
    run="results/car_vgrid_19016936_laser12_cap3_s${seed}"
    test -s "$run/CarAgent.onnx" && test -s "$run/configuration.yaml" || { echo "Missing $run"; break; }
done

# One pair, one map, deterministic; inspect its CSV before the full array.
sbatch --array=0 --export=ALL,TRAFFIC_SCENARIO=0 traffic_all_seed_pairs_hpc.sh

# Once the smoke test succeeds, run all 50 tasks / 500 map rollouts.
sbatch traffic_all_seed_pairs_hpc.sh
```

For the smoke test, task 0 is seed 0 in both seats, deterministic, scenario 0. In the full array, task `2*(5*seat0+seat1)` is deterministic and the next task is stochastic. A completed full run has 50 `traffic.csv` files, each with a header and ten rows, and 50 `traffic_summary.csv` files. Stochastic mode is only one sampled rollout per pair/map, so repeat promising or surprising pairings before drawing statistical conclusions.

Check the smoke result at `traffic_results/pairwise_<job-id>/s0_s0_deterministic/traffic.csv`: its single row must include `finish_tile0/1` and all ten `tile_seconds...` columns. A timeout's tile-time values add up to approximately 100 simulated seconds (5000 steps at the current 0.02-second fixed timestep); a goal or collision stops that car's tile clock earlier. The `_summary.csv` also counts timeout finish tiles and reports average time on each tile for timed-out cars.

## Full held-out map evaluation with a changeable car count

This is a **separate scene and Linux build** from the ten-case pilot above. The manifest
`Assets/Resources/Traffic/traffic_validation_maps.json` lists the same 100 sorted
`VoronoiVal` maps used by single-car evaluation, but **does not prescribe routes**.
On each episode the scene samples new spawn/goal points from the largest connected
Asphalt component on the selected map. Endpoints are at least 8 m apart from every
other endpoint, each car's own spawn and goal are at least 20 m apart, and all
endpoints are two cells inside the map boundary. These are lower bounds, not fixed
distances. The ten-case pilot still uses its fixed routes. The full scene has a
ten-car pool; `TRAFFIC_CARS=2..10` selects how many are active without rebuilding.
Placement is seeded, so deterministic and stochastic evaluations use the same
coordinates for a given map/rollout, and the CSV records those coordinates and seed.
One random placement per map is the default; `TRAFFIC_ROLLOUTS_PER_MAP=3` tests three.

The manifest is reproducible from:

```bash
python hpc/traffic/generate_validation_map_manifest.py
python hpc/traffic/check_random_placement.py
```

In Unity, wait for compilation, then choose **Tools > Traffic > Create all-validation
traffic scene**. This copies `TrafficTwoCar.unity` to
`Assets/Scenes/TrafficAllValidation.unity` without changing the pilot, assigns the
100-map manifest, and creates a pool of ten cars with separate goals and behavior names.
It starts with two active cars in the Editor. The desired default active count is on
`TrafficScenarioManager`; the HPC evaluator overrides it at process startup with
`-traffic-car-count`. The minimum spacing and placement-attempt count can be edited
on the manager in the Unity Inspector. The route sampler always generates locations
for the full ten-car pool, so the first routes for a given seed remain the same when
comparing different active car counts. More than ten cars would need a larger pool,
an offline geometry check, and a rebuild.

Build **only** `TrafficAllValidation.unity` as a separate Linux player named
`car.x86_64`, under `car1/Linux_traffic_all_validation/`. Confirm its `build_info.txt` says
`multi-car-random-placement-v4`, `Traffic car pool size: 10`,
`Traffic placement mode: random-connected-asphalt`, and
`Traffic scenario manifest: traffic_validation_maps`. The trained 77-observation ONNX
models and YAML configurations are unchanged; no retraining is needed. Preserve the
old `Linux_traffic/` build for the ten-case pilot.
Keep all ten `Behavior Parameters` in **Default** mode so Python can supply actions;
`Inference Only` would silently bypass the evaluator.

Upload the complete new build directory and these files to `car1/` on HPC:

| Local project file | HPC location |
| --- | --- |
| `hpc/traffic/eval_traffic.py` | `car1/eval_traffic.py` |
| `hpc/traffic/traffic_full_validation_hpc.sh` | `car1/traffic_full_validation_hpc.sh` |
| `Assets/Resources/Traffic/traffic_validation_maps.json` | `car1/traffic_validation_maps.json` |
| `eval_checkpoints.py` | `car1/eval_checkpoints.py` (unchanged, if already present) |

The job uses the existing `car1/mlagents.sif` and cap-3 seed models under
`car1/results/car_vgrid_19016936_laser12_cap3_s0` through `_s4`.
On HPC, start with one map in both action modes, then submit all 100 maps:

```bash
cd /d/hpc/home/kk42117/mag/car1
mkdir -p logs
chmod +x Linux_traffic_all_validation/car.x86_64
sed -i 's/\r$//' traffic_full_validation_hpc.sh
sbatch --array=0-1 --export=ALL,TRAFFIC_CARS=2,TRAFFIC_MODEL_SEEDS=0,TRAFFIC_SCENARIO=0 traffic_full_validation_hpc.sh

# After checking both one-row CSVs and their map-index/car-count columns:
unset TRAFFIC_SCENARIO
sbatch --export=ALL,TRAFFIC_CARS=2,TRAFFIC_MODEL_SEEDS=0 traffic_full_validation_hpc.sh
```

`TRAFFIC_CARS=3` through `10` uses the same ten-car build. The placement seed is
the same across car counts, but the vehicles interact differently once moving.
Run the one-map smoke submission with each new car count before its full sweep;
confirm that the CSV has outcomes for every requested seat and reports the expected
`num_cars`, `map_index`, `random_placement=1`, and Asphalt spawn/goal coordinates.
`TRAFFIC_MODEL_SEEDS=0` broadcasts seed 0 to every seat; `0,1,2` assigns seeds to
three seats in order. Every submission has 20 array tasks: ten ten-map shards in
deterministic and stochastic mode, limited to six concurrent tasks. Results go to
`traffic_results/full_validation_<job-id>/carsN_seeds.../<mode>/shardK/` with one
`traffic.csv` and `traffic_summary.csv` per shard. The CSV records a separate outcome,
reward, finish step/tile and tile-time vector for every active car, plus the actual
map index and minimum pairwise car separation. Do not combine pilot and full-validation
rows as though they sampled the same scenarios. Random placement tests can be
reproduced by keeping `TRAFFIC_PLACEMENT_SEED` (default 3401) and the same map and
rollout index; changing it gives another placement draw. The 8 m/20 m spacing
guarantees do not imply that routes intersect, so a lack of car encounters on a
map is an expected possible outcome, not an evaluation failure.

### Matched 2/5/10-car seed-mix comparison (250 episodes per action mode)

Car count is selected by `TRAFFIC_CARS`/the Unity `-traffic-car-count` argument,
**not** by the training YAML. This keeps the same built scene and physics for all
comparisons. Submit the six conditions below after a one-map smoke test. Each job
has ten map shards × two action modes, 25 episodes per shard, hence **250 episodes
per mode per condition**. In each shard every map is sampled two or three times;
the 50 third placements over all shards are assigned by a fixed shuffled schedule
(one extra map per trained seed in each shard, yielding 50 episodes per seed).
The same map, placement seed and trained-model assignment are used in deterministic
and stochastic modes. The `same` and `mixed` conditions share the seed in seat 0.

| Cars | `same` | `mixed` |
| --- | --- | --- |
| 2 | One random seed per map, used in both seats | Two distinct seeds per map |
| 5 | One random seed per map, used in all seats | One car from each of the five seeds |
| 10 | One random seed per map, used in all seats | Two cars from each of the five seeds |

In `mixed`, seat 0 keeps the seed chosen for that map's `same` condition; the other
seeds are assigned deterministically from `TRAFFIC_ASSIGNMENT_SEED` (default 7319).
Across 100 maps, each seed anchors exactly 20 maps. The assignment is recorded in
the per-seat `model_seedN` CSV columns. For the two-car mixed condition, each ordered
pair of different seeds appears on five maps. The five
seed models are the existing cap-3 laser77 models, not new training runs.

```bash
cd /d/hpc/home/kk42117/mag/car1
mkdir -p logs
chmod +x Linux_traffic_all_validation/car.x86_64
sed -i 's/\r$//' traffic_full_validation_hpc.sh

# Smoke-test one map, both deterministic and stochastic, for all six conditions.
for cars in 2 5 10; do
  for strategy in same mixed; do
    sbatch --array=0-1 --export=ALL,TRAFFIC_CARS="$cars",TRAFFIC_SEED_STRATEGY="$strategy",TRAFFIC_SCENARIO=0 traffic_full_validation_hpc.sh
  done
done

# Once all six smoke tests have valid CSVs, run 250 episodes per mode/condition.
unset TRAFFIC_SCENARIO TRAFFIC_EPISODES TRAFFIC_ROLLOUTS_PER_MAP
for cars in 2 5 10; do
  for strategy in same mixed; do
    sbatch --array=0-19%2 --export=ALL,TRAFFIC_CARS="$cars",TRAFFIC_SEED_STRATEGY="$strategy" traffic_full_validation_hpc.sh
  done
done
```

The full sweep is six Slurm array jobs (120 array tasks total), not one job per
episode. The `%2` limit allows at most two tasks from each array at once (up to
12 across all six arrays). If testing needs a second independent draw, change `TRAFFIC_PLACEMENT_SEED`
and `TRAFFIC_ASSIGNMENT_SEED` deliberately and record both values. Each CSV reports
the model seed in every seat, actual spawn/goal coordinates, map, action mode,
outcomes, and minimum endpoint distance.

For **future** runs, when one scheduler job is preferable, upload
`hpc/traffic/traffic_full_validation_one_job_hpc.sh` as
`car1/traffic_full_validation_one_job_hpc.sh` and submit it once:

```bash
cd /d/hpc/home/kk42117/mag/car1
mkdir -p logs
sed -i 's/\r$//' traffic_full_validation_one_job_hpc.sh
sbatch traffic_full_validation_one_job_hpc.sh
```

This requests one 64-CPU/128-GB, 24-hour Slurm allocation and runs the same six
conditions, map shards and deterministic/stochastic episodes as the array sweep,
up to eight evaluator processes at a time inside that allocation. It therefore
produces 250 episodes per action mode per condition on the 100 held-out maps,
with 2, 5 or 10 interacting cars per episode, and keeps both modes paired on
the same map/placement/model draw. Outputs for all six conditions go under one
`traffic_results/full_validation_<job-id>/` folder. The existing
`check_full_validation.py --expected full --require-six-conditions --job-ids <job-id>`
checks all six condition subfolders and reports any missing condition. Unlike
the post-training single-car evaluator, the
traffic evaluator restarts Unity for each joint multi-car episode and currently
does not pool episodes across many persistent Unity environments. The new job
packaging does not change that sampling or physics logic. Do not submit this
one-job version while the equivalent six-array sweep is still running.

To check downloaded/completed smoke tests without pasting every CSV, upload
`hpc/traffic/check_full_validation.py` as `car1/check_full_validation.py` and run:

```bash
cd /d/hpc/home/kk42117/mag/car1
python3 check_full_validation.py --expected smoke --job-ids 19375063
```

For the six full jobs, replace `smoke` with `full`. The checker reports per-mode
row/file counts, outcome totals, seed-mix correctness, Asphalt placements, and
whether deterministic/stochastic episodes use the same maps, placements and models.
`PASS` verifies the CSV data; also confirm Slurm tasks are `COMPLETED` with exit
code `0:0` using `sacct`.

If the six full jobs are the newest six `full_validation_*` result folders, check
them together with `python3 check_full_validation.py --expected full --latest 6`.
Otherwise use `--job-ids` followed by their six array-job IDs to avoid including
smoke-test folders.

### Shared-policy multi-car training and paired evaluation

Do **not** train on `Linux_traffic_all_validation`: its manager deliberately uses
held-out `VoronoiVal` maps and separate `TrafficCar0`... behavior names. The new
`TrafficTraining.unity` scene uses all 400 `VoronoiTrain` maps, random separated
Asphalt starts/goals, and one shared `CarAgent` behavior for its ten-car pool.
The training array activates 2, 5 or 10 cars in separate runs. Finished cars
stay parked until the shared episode ends, but no longer contribute repeated
no-op decisions to PPO.

In Unity, use **Tools → Traffic → Create shared-policy training scene**. The
manifest `Assets/Resources/Traffic/traffic_training_maps.json` is already in this
project; if the map pool changes, regenerate it with
`python hpc/traffic/generate_validation_map_manifest.py --maps-dir Assets/Resources/Maps/VoronoiTrain --out Assets/Resources/Traffic/traffic_training_maps.json --expected-count 400`.
In Build Settings enable **only** `Assets/Scenes/TrafficTraining.unity`, and build
Linux into `Linux_traffic_training`. Confirm its `build_info.txt` says
`Traffic map source: VoronoiTrain`, `Traffic map selection: random-each-episode`,
all ten seats `CarAgent`, and matched 77 observations. Keep the existing
`Linux_traffic_all_validation` scene for evaluation (never train on it). For
trajectory recording, **rebuild that validation scene** into
`Linux_traffic_all_validation` after the `TrafficScenarioManager.cs` and
`BuildInfoWriter.cs` changes here. Its new `build_info.txt` must contain
`Traffic trajectory schema: per-seat-xz-step-v1`. The training build does not
need to be rebuilt merely for the optional trajectory recorder.

Upload the **entire** new `Linux_traffic_training` build directory to
`/d/hpc/home/kk42117/mag/car1/`, plus these files at the `car1` root:

- `hpc/traffic/traffic_multiagent_train_hpc.sh` → `traffic_multiagent_train_hpc.sh`
- `hpc/traffic/traffic_multiagent_eval_hpc.sh` → `traffic_multiagent_eval_hpc.sh`
- `hpc/traffic/prepare_multiagent_config.py` → `prepare_multiagent_config.py`
- `hpc/traffic/select_multiagent_checkpoint.py` → `select_multiagent_checkpoint.py`
- `hpc/traffic/check_multiagent_training.py` → `check_multiagent_training.py`
- `hpc/traffic/summarize_multiagent_checkpoints.py` → `summarize_multiagent_checkpoints.py`
- `hpc/traffic/plot_multiagent_trajectories.py` → `plot_multiagent_trajectories.py`
- `Assets/Resources/Traffic/traffic_training_maps.json` → `traffic_training_maps.json`

Also upload the updated `hpc/traffic/eval_traffic.py` as `car1/eval_traffic.py`,
the **entire rebuilt** `Linux_traffic_all_validation` directory, and these ten
text maps from `Assets/Resources/Maps/VoronoiVal` into `car1/VoronoiVal`:
`curved_401`, `406`, `407`, `438`, `442`, `444`, `451`, `480`, `484`, and `494`
(all with `.txt` extensions). The job checks that each map is present before
the final-checkpoint trajectory evaluations. The existing
`traffic_validation_maps.json`, `mlagents.sif`, `stats_patched.py`, and
`trainer_controller_patched.py` must still be present. Each of the five original
`results/car_vgrid_19016936_laser12_cap3_s0`...`s4` directories needs its
`configuration.yaml` and `CarAgent/checkpoint.pt`. The final `.onnx` alone is
not sufficient to fine-tune. The training script checks all of this before
launching. The normal `car_config.yaml` is **not** edited; each task derives a
matched config from its seed's saved configuration.

First smoke-test the new Unity build and fine-tune initialization at **each**
density (tasks 0, 10 and 20; seed 0; 300,000 steps). From the cluster's `car1`
directory:

```bash
cd /d/hpc/home/kk42117/mag/car1
mkdir -p logs configs
sed -i 's/\r$//' traffic_multiagent_train_hpc.sh traffic_multiagent_eval_hpc.sh
chmod +x Linux_traffic_training/car.x86_64 Linux_traffic_all_validation/car.x86_64
sbatch --array=0,10,20 --export=ALL,TRAFFIC_TRAIN_STEPS=300000 \
  traffic_multiagent_train_hpc.sh
```

Check that all three tasks are `COMPLETED 0:0`, have no placement/observation
errors, and each produced `CarAgent.onnx`. Then submit full training:

```bash
cd /d/hpc/home/kk42117/mag/car1
TRAIN_JOB=$(sbatch --parsable traffic_multiagent_train_hpc.sh)
echo "Training array: $TRAIN_JOB"
```

Training is one 30-task array. Tasks 0–9 use two cars, 10–19 use five, and
20–29 use ten; within each block, the first five fine-tune source seeds 0–4
from their `.pt` checkpoints and the next five train scratch models with
matching PPO settings. Up to four run concurrently. Fine-tunes get **20M
additional** shared-policy agent steps (40M lifetime including the original
20M single-car pretraining); scratch models get **40M** multi-car steps. Both
save ONNX checkpoints every 1M steps and retain all 20 or 40 milestones.
To keep the simultaneously active car count
near 60–64 per task, the script uses 32 Unity environments for 2 cars, 12 for
5 cars, and 6 for 10 cars. Thus the environment count also differs between
training-density groups; do **not** claim that a difference is caused solely
by traffic density. Fine-tunes initialize weights in new optimizer/trainer
runs, rather than resuming the old single-car run exactly. All 30 models use
the same 77-observation laser schema.

The training allocation is 24 hours. **40M steps may need more than one
allocation**, especially with ten cars. If a task times out after saving a
checkpoint, resume only the unfinished task IDs, retaining the *original*
training array ID in the run names. For example, replacing `5,15,25` with the
actual unfinished task IDs:

```bash
sbatch --array=5,15,25 --export=ALL,TRAIN_ARRAY_ID=$TRAIN_JOB,TRAFFIC_RESUME=1 \
  traffic_multiagent_train_hpc.sh
```

Use the same `TRAIN_JOB` value even if several resume allocations are needed.
The resume path uses ML-Agents `--resume` with the run's existing config and
checkpoint; it does not reinitialize from the original single-car model. Never
use `--force` here. Before launching evaluation, verify completion:

```bash
python3 check_multiagent_training.py --results-dir results \
  --train-array-id "$TRAIN_JOB"
```

This checks all 30 final ONNX files and all **900** distinct million-step
checkpoints. Do not prune intermediate `.onnx` files before evaluation.

Evaluation is separate from training; saved checkpoints are evaluated after
training, rather than pausing PPO every million steps. Once the checker passes,
smoke-test checkpoint resolution and the evaluator on one trained checkpoint
and one original baseline:

```bash
sbatch --array=0,900 --export=ALL,TRAIN_ARRAY_ID=$TRAIN_JOB,TRAFFIC_EVAL_CARS=2,TRAFFIC_EVAL_EPISODES=1 \
  traffic_multiagent_eval_hpc.sh
```

When both smoke tasks succeed, submit **three** full instances of the
evaluation array, one for each test density:

```bash
for cars in 2 5 10; do
  sbatch --export=ALL,TRAIN_ARRAY_ID=$TRAIN_JOB,TRAFFIC_EVAL_CARS=$cars \
    traffic_multiagent_eval_hpc.sh
done
```

Each evaluation array has 905 tasks, at most two active at once: 900 tasks
for trained checkpoints (five seeds × [20 fine-tune + 40 scratch] × three
training densities), plus five original-model baselines. Each task evaluates
one model in **both deterministic and stochastic modes**, 250 episodes per
mode, across all 100 held-out maps. Together the three arrays produce
**5,430 CSV pairs of metrics/summaries and 1,357,500 joint episodes**
(1,350,000 from checkpoints, 7,500 from original baselines). This is a very
large experiment; allow substantial cluster time and check local array/job
limits before submission. As before, all cars on a map share the same model.
The **final** milestone of each of the 30 trained runs, plus the five original
baselines, additionally gets ten-map trajectory evaluation at each car count:
one deterministic and five stochastic rollouts per map, **6,300 additional
joint episodes** across the three car-count arrays. These are the same ten
held-out maps as the single-car trajectories, with all seats' positions sampled
every five physics steps. On each map every model, action mode and rollout uses
the same seeded random Asphalt routes; the 2/5-car routes are prefixes of the
10-car placement. The routes are not identical to the old *single-car* fixed
routes. Each final result has `deterministic/trajectories/trajectories.csv`,
`stochastic/trajectories/trajectories.csv`, and a ten-panel
`trajectory_grid.png` showing the map and all car paths. Intermediate
checkpoints get the 250+250 evaluation but no per-step trajectory file, to
keep storage manageable.
If the cluster rejects a 905-task array, submit the same script in nonoverlapping
`--array=0-299%2`, `--array=300-599%2`, and `--array=600-904%2` ranges for
each `TRAFFIC_EVAL_CARS` value; the output paths remain the same.
The 2/5/10-car test conditions use paired map and placement schedules at
each checkpoint. Results are under
`traffic_results/multiagent_checkpoints_<TRAIN_JOB>/train{0,2,5,10}/...`.

When all three arrays have completed, run on HPC:

```bash
python3 summarize_multiagent_checkpoints.py \
  "traffic_results/multiagent_checkpoints_$TRAIN_JOB"
```

It checks all expected summary counts and creates
`checkpoint_curve_by_seed.csv`, `checkpoint_curve_means.csv`, and
`paired_delta_vs_original_by_seed.csv`. The milestone is in millions of **new**
multi-agent steps; `lifetime_million` includes the prior 20M single-car steps
for fine-tunes, and the CSV also records the actual saved checkpoint step.
The old `summarize_multiagent_comparison.py` is only for the earlier final-only
evaluation layout and is not used by this checkpoint sweep. Confirm Slurm
`COMPLETED/0:0` for all task IDs before interpreting the curves.

### Optional separate vehicle-distance lasers (future 89-observation experiment)

`LaserTileSensor.includeVehicleDistanceObservations` is off in the existing
77-observation scenes. In that mode, cars retain their old Terminal-tile reading.
When enabled, each of the 12 directions adds a sixth value: distance to the nearest
other vehicle, normalized as `d/(d+S)` with `S=10 m` by default and `1` for no
vehicle. Floor-tile readings ignore vehicles. This gives `17 + 12*6 = 89`
observations. The scale can be changed with `laser_vehicle_distance_scale_m`;
it does not limit how far the laser detects cars.

In Unity, **Tools → Traffic → Create vehicle-distance training scene** copies
`TrafficTraining.unity` to `TrafficTrainingVehicle.unity`. The matching validation
menu copies `TrafficAllValidation.unity` to `TrafficAllValidationVehicle.unity`.
Both set all ten cars to 89 observations and enable the vehicle channel. These
menu actions leave the original 77-observation scenes untouched. Build either
variant as the *only* enabled scene in Build Settings, into a distinct folder.
For its YAML set `laser_vehicle_distance_observation_enabled: 1` and
`laser_vehicle_distance_scale_m: 10.0`. The evaluator accepts
`--expected-observations 89`. A run with the wrong observation size fails with
an explicit error; a 77-input checkpoint cannot directly drive the 89-input
build. The training/evaluation arrays above intentionally continue using 77.
