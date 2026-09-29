# TODO

## HPC experiment conventions

- Use 64 training environments with 64 allocated CPUs for new sweeps. With the current PPO setup, use `buffer_size: 16384` so there are still 256 experiences per environment per update. Limit each Unity process to one internal job worker and numerical-library thread pools to one thread; otherwise colocated 64-environment jobs can exhaust the node's thread limit. Use 32 environments for checkpoint evaluation unless a separate benchmark demonstrates an advantage from more.
- Apply each run's saved YAML environment parameters during checkpoint and trajectory evaluation, including the vehicle speed cap; do not evaluate capped policies under scene-default uncapped dynamics.
- Reserve ML-Agents port blocks atomically per node in new Slurm scripts to prevent array-task worker/port collisions.
- Preserve the original 8-ray/57-observation laser build for its checkpoints. Build the 12-ray/77-observation version separately; models cannot be exchanged between these observation sizes.

## Vehicle dynamics

- Replace the legacy `100000 N·m`-per-wheel SpeedLimited/Gravel brake with calibrated, surface-aware braking that reaches the tile speed quickly without wheel lock or skid.


- Create a Lagrangian penalty for near termination based on the current speed

## Laser observations

- Test denser forward laser coverage (add rays near +/-12.5 degrees) before combining it with a small set of fan points.
- Test nonlinear distance normalization `d / (d + S)` instead of division by the roughly 140 m map diagonal.
  - For special tiles, compare a fixed `S = 20 m` with `S` equal to the conservative asphalt stopping distance at `EffectiveTopSpeedMS`. This remains fixed for a configured run rather than changing with current speed.
  - Keep road-boundary normalization local and independent of top speed. The 500 exported maps have an effective median road width of about `6.2 m`, so use the approximately `3 m` half-width as `S` and verify it experimentally.
  - Determine the stopping-distance scale using the measured braking test and include control delay plus a safety margin; do not assume theoretical 1 g braking.
- Test a speed-aware braking-margin observation in addition to, rather than instead of, the fixed-normalized laser distances. Use velocity projected along each ray and an empirically measured safe braking deceleration.
- Treat laser density, distance normalization, and braking-margin observations as separate ablations.

## Future road-network maps

- Explore existing CARLA/ASAM OpenDRIVE maps, OpenStreetMap-to-OpenDRIVE conversion, and SUMO-generated networks as sources of more realistic layouts.
- Convert selected road networks offline into the current tile-grid format. First target approximately 200 x 200 m at the existing 1.5 m cell resolution; later consider 300-500 m.
- Before increasing grid resolution, replace per-cell tile GameObjects/colliders with a compact ground collider and combined or chunked rendering, while retaining the logical tile-type grid for sensors and rewards.
- For larger road networks, use connected-road spawn/goal selection, route-distance-based curriculum and episode budgets, bounded laser range, and reassess time-penalty scaling.

## Traffic after the two-car pilot

- Compare existing 3 m/s laser77 checkpoints on the fixed ten two-car cases, then compare fine-tuning the best single-car checkpoint against training a traffic policy from scratch.
- Replace fixed second routes with dynamic connected-road placement that enforces minimum spacing and produces likely encounters. Extend scenario generation and evaluation beyond two cars.
- Test whether the 15-degree forward laser spacing misses other cars. If necessary, add a distinct moving-vehicle reading or wider forward detection; either change requires a new observation shape and training build.
- In a separate ablation, make static Terminal contact use the physical chassis footprint rather than only the car's center. Keep the legacy center rule for the first traffic-transfer comparison.
