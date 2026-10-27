# Multi-agent traffic recovery curriculum plan

Purpose: improve the **deterministic** policy's ability to escape a blocked
route, yield, reverse, and then resume goal-directed driving.  Use the current
89-observation policy, including the separate vehicle-distance rays, 3 m/s
cap, existing collision/terminal penalties, and a 5,000 shared-step limit.

This is a 7.5M-step pretraining/fine-tuning phase followed by ordinary 10-car
traffic training.  It is intentionally **not** a replacement for full-map
training.  After 7.5M, run normal random 10-car maps for the remaining budget;
retain roughly 10--15% recovery cases so reverse/yield behaviours are not
forgotten.

## General randomisation rules

- Randomise spawn position along its valid lane, goal distance, and small
  heading jitter (about plus/minus 10--15 degrees).
- Use left/right mirror and 90-degree rotated versions of each topology.
- Keep early headings approximately road-aligned.  A fully arbitrary 360-degree
  initial heading mixes a separate orientation-recovery task into the first
  driving lesson and makes it harder to diagnose learning.
- Goals must always be on reachable Asphalt.  A goal placed beyond an
  unbroken Terminal wall is impossible and teaches nothing useful.
- Randomise timing/headway and approach offsets in multi-car stages.  Never
  create a perfectly symmetric deterministic four-car situation every time:
  identical observations plus one shared policy can produce a permanent
  symmetry deadlock.

## Stage 1 -- one-car road following, 0M--2M

All routes begin short and expand linearly within their own substage.  The
agent is initially facing approximately along the road toward its goal.

| Training steps | Maps | Route lesson |
|---|---|---|
| 0--200k | `recovery_01_straight` | Straight route, roughly 8m to 50m. |
| 200k--500k | `recovery_07_s_curve` | Short, almost-straight routes expand through the first bend and then the full S. |
| 500k--800k | `recovery_02_turn` | Goal expands from the straight approach through the 90-degree corner. |
| 800k--2M | All three | Uniform/random mixture, random road direction, route length and small heading jitter. |

The corner and S-route are useful: they teach steering from road geometry,
rather than merely accelerating toward a directly visible goal.

## Stage 2 -- one-car dead-end recovery, 2M--3.5M

| Training steps | Case | Required response |
|---|---|---|
| 2.0--3.0M | `recovery_06_blind_alley` | Spawn near the capped branch, facing the Terminal cap; goal is on the main road behind/side of the car. Reverse out, turn, then proceed. |
| 3.0--3.5M | New dead-end-*detour* layout | Goal is spatially forward but reachable only through a side loop/bypass. Directly ahead is Terminal, so reversing first is necessary even though it initially appears to move away from the goal. |

The second half needs a new map; the existing blind alley alone cannot put a
reachable goal beyond its closed Terminal cap.

## Stage 3 -- one learner plus static visible vehicle, 3.5M--4.5M

Use the passing-bay geometry and a kinematic/static car obstacle.  It must be
included in vehicle ray casts and normal collision physics, but must not be an
ML-Agent seat, receive rewards, or terminate the shared episode by itself.

| Training steps | Case |
|---|---|
| 3.5--3.7M | Fixed parked vehicle; large bypass/passing space; learner practices perceiving a vehicle and moving around it. |
| 3.7--4.3M | Parked vehicle; goal is beyond the blocker and requires reverse/turn/bypass. |
| 4.3--4.5M | Same blocked cases, plus 20% unobstructed/easy routes. |

The 20% easy cases prevent a policy that reverses whenever it sees a vehicle.

## Stage 4 -- two moving cars, crossroads, 4.5M--5.5M

Use `recovery_04_crossroads`.

| Training steps | Case |
|---|---|
| 4.5--5.5M | Random crossing paths and independent start/heading offsets; goals remain reachable without reversing in most cases. |

This separates basic yielding/braking from the harder head-to-head reverse
lesson.

## Stage 5 -- two-car head-to-head passing bay, 5.5M--6.5M

Use `recovery_05_passing_bay`.  Both goals lie beyond the opposing car.  One
car must create space by reversing into the bay, then continue after the other
passes.

| Training steps | Case |
|---|---|
| 5.5--6.5M | Opposing start/goal pairs with independent heading jitter; one car must create passing space. |

## Stage 6 -- four-car intersection/gridlock, 6.5M--7.5M

Use the crossroads, with four cars approaching from the four arms and goals
on the opposite arm.  Every car initially sees another vehicle ahead.

| Training steps | Case |
|---|---|
| 6.5--7.5M | Four cars with small independent approach and heading offsets, preventing symmetric deadlock. |

The small asymmetries are essential.  A perfectly symmetric four-way face-off
with one shared deterministic model gives each car the same observation and
action, so it has no principled way to choose a yielding car.

## Implemented curriculum mode

`TrafficScenarioManager` now exposes `recoveryCurriculumEnabled` (or the
environment parameter `traffic_recovery_curriculum_enabled=1`) and
`recoveryCurriculumEndSteps` (default 7.5M). While active it uses
`Maps/TrafficRecovery`, stage-specific routes, one/two/four active seats, and
small independent heading/spawn offsets. It then automatically returns to the
normal configured traffic map source and route setup.

`recovery_08_dead_end_detour` and the kinematic `TrafficVehicleObstacle` are
implemented. The parked obstacle participates in vehicle laser intersections
and ends a colliding learner with `VehicleCollision`, but is not an agent seat.
The recovery map loader filters out its JSON catalogue, keeping map indices
stable. Road-direction variants are implemented by reversing routes and
randomising offsets; explicit rotated/mirrored map files remain a possible
future expansion.
