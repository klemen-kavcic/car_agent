# Traffic recovery curriculum maps

These eight 66x66 maps are intentionally stored separately from `VoronoiTrain`
and `VoronoiVal`. Adding them directly to either existing folder would change
the sorted map indices used by completed experiments and manifests.

Tile codes follow the normal Voronoi maps: `0` gravel, `1` asphalt, and `4`
terminal. Asphalt is the drivable road centre, gravel is a forgiving low-speed
shoulder, and terminal is the hazard outside the road. The asphalt centre is
targeted to four cells wide, the modal straight-road width in `curved_1` and
the other 66x66 curved maps; rasterised bends may vary by one cell.

The map shapes alone do not make reversing necessary. The recovery scenario
builder must define the spawn/goal pairs and, for later stages, place blocked
or opposing cars in the intended lanes. The catalog records each map's purpose.

`recovery_06_blind_alley` uses a terminal-walled, two-cell-wide (3 m) final branch:
the car begins facing the cap and cannot safely turn around using gravel
shoulders, so it must reverse back to the main road. `recovery_08_dead_end_detour`
then adds the harder case: the goal is spatially forward of the start, but the
direct upper branch is capped by Terminal. The only valid route is to reverse
to the main road and take the lower bypass.

The recovery sequence also uses `recovery_04_crossroads` for two distinct
four-car lessons: a long-approach crossroads and a final close-range deadlock.
In the deadlock lesson, four cars begin near the junction facing goals beyond
the opposing cars. Continuing straight is unsafe; reversing or yielding is
needed before re-entering the junction. Spawn lanes receive small offsets and
heading jitter on every episode.

`recovery_09_circulation_loop` is a separate four-car close-range negotiation
lesson on an open asphalt plaza. Cars start close together in a cyclic layout,
with each goal forward on clear asphalt. It deliberately does not replace the
head-to-head deadlock: this lesson isolates response to nearby vehicles from
road-edge following and does not hide a car's goal directly behind another
vehicle.

`TrafficScenarioManager` selects this folder only while its optional recovery
curriculum is enabled (or its matching environment parameter is set); normal
training and evaluation never use these maps.

Regenerate maps and PNG previews with:

```powershell
python 'Map Gen/generate_recovery_maps.py'
```
