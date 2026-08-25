- Create car
    - body
    - wheels (wheel colliders)
    - script for handling the car
    - script for turning wheels
- Added MLAgents
    - write code to expose inputs
    - agent (with reward function)
    - behaviour parameters
    - decision requester
    - problem with crashing, export exe, run without graphics
- run a test
    - 2 million steps, 13200 seconds
    - put the agent into behavior parameters, inference only, it finds the end point
    - tries to hit it sideways, probably extra reward for hitting from the front
- hpc
    - export linux server version
    - upload to hpc
    - create apptainer for running it on arnes hpc
    - test runs on 2m (8200s) and 20m steps
- 0621
    - fixed steering, now can change 60degrees in a second (not instant)
- 0622
    - added different tiles, termination, low traction, low high speed
    - made a mistake and it didnt reset the board so the results are taught for a particular map, 2m worked badly, 20m worked better but it just straight lined to the goal, ignoring different types
    - board now resets everytime
    - changed the reward funtion
        - extra reward for straight to the goal
        - higher time penalty
        - lower distance delta reward (longer path sometimes better)
    - optimise learning on the hpc 8 at the time
- 0627
    - more limit on top speed on grass
    - limit speed backwards
    - penatly for starting to tip over (square number of wheels * penalty)
    - ice very low friction for now
    - added lidar-esque sensors
    - added perling noise for map generation
        - perling noise for the 3, terminal additionally with noise
        - start and stop always on normal
    - lowered center of gravity

- 0706
    - new model
    - tesla model 3
    - weight 1800
    - separate torque front, back (1500, 1000)
    - separate gas/brake pedals
    - separate gear for reverse (has to be still and holding brake)
    - regen braking when no gas is pressed
    - brake balance



- 0807 2pedal
    - removed map orientation observation (leftover from before)
    - fixed the sensors, out of bounds is now considered terminal
    - added observation of current gas/brake
    - observation space 3 continuous, 1 discrete
    - added posibility to degrade some of the penalties
    - didnt learn well
    - with reward multiplying 0.01 didnt work at all, worked with 0.0008
    - possibility to have first few runs spawn goal closer to the ca
- 0807 1pedal 2_2
    - observation space 2 continuous, 2 discrete
- 0807 1pedal 2_1
    - even simpler, 2continuous 1 discrete
    - decided to go with 2_2 (more realistic and it looked promising)

- 0715
    - each step 0.02s (decision every 5 steps, every 0.1s)
    - change steering (instead of desired angle it is delta, how much it wants to change) 35/s - 3.5 degrees per second and divide it by 5 per step
    - changed reward for coming closer - 100 parts how closer it got
    - added angular speed observation to make it markov complete
    - -15 penalty doesnt work, sotps moving

| # | Observation | Size | Source |
|---|---|---|---|
| 1 | Direction to goal (local space, normalized) | 3 | `transform.InverseTransformDirection(toGoal.normalized)` |
| 2 | Distance to goal | 1 | `toGoal.magnitude` |
| 3 | Velocity (local space) | 3 | `transform.InverseTransformDirection(rb.linearVelocity)` |
| 4 | Yaw rate (local space) | 1 | `transform.InverseTransformDirection(rb.angularVelocity).y` |
| 5 | Current steering angle, normalized | 1 | `currentSteerAngle / maxSteerAngle` → roughly [-1, 1] |
| 6 | Reverse gear flag | 1 | `carController.reverseGear ? 1 : 0` |
| 7 | Brake-pedal-selected flag | 1 | `pedalIsBrake ? 1 : 0` |
| 8 | Pedal magnitude | 1 | `gasInput + brakeInput` (only one is ever nonzero) |
| **Base subtotal** | | **8** | |
| 9 | Current tile type, one-hot | 4 | Normal / Slippery / SpeedLimited / Terminal |
| 10 | Grid sensor readings, one-hot per point | 37 × 4 = 148 | `CarSensor` — stadium-shaped pattern, radius 3, sensor spacing 3m |
| **Total** | | **112** | |                      24


    - higher penalty later after it is more successful
    - different sensor placement to make them further out
    - checked it uses ADAM

    - remove brakeInput as condition for reverse
    - fixed clamping for pedal press
        - mlagents cover the raw outputs from neural netwroks, nn outputs a value and a distribuition, then magents choose a value and scale it to -1 - 1, so i scaled and then clamped the pedal press to make it 0-1, before there was a bug; ContinuousActions[1] is a raw Gaussian sample clipped by ML-Agents to [-1, 1]
    - training cirriculum for the first few steps
    

    Referencing the hpc folder without copying: Yes, this is possible — Unity's Project window only ever shows Assets/, but you can create a symlink/junction inside Assets/ that points at an external folder, and Unity will follow it and show the contents as if they were really there, with no physical duplication on disk. On Windows:


    mklink /J "Assets\Models\HpcResults" "D:\path\to\your\local\results\folder"

0729
    - checked if observations and inputs are in correct order
    - hopefully eliminated the unnacounted for episodes, the counter counted just action steps not all of them, i think all of the unnacounted for were timeouts
    - make random seeds
    
0802
    - added code for testing partially made models
    - started the new maps
    - generated pool of train and eval maps
    - implemented changeable penatly for termination, for max step and adaptive max step
    - lagrangian for termination the only one that has better results
    - max step lagrangian has significantly worse results - remove it
    - adaptive max step doesnt show any improvements - removed
    - lagrangian for termination stays
    - added stat for how much time agent spends on a given surface (problem for slow surfaces - more time)
    

0821
    - added stat how many times it switches surface on the map
    - added std deviation that was missing before
    - removed the two lagrangians that did not work
    - added 10 fixed maps with fixed start/end to compare the different trajectories
    how wide is the road?


 this is essentially potential-based reward shaping (Ng, Harada & Russell 1999): define a "danger potential" Φ(s) = (terminal sensors currently triggered)/7, and reward the change in it each step (−ΔΦ). That framing matters because it's the one variant of reward shaping that's provably policy-invariant — a full round trip (approach then fully retreat) always nets to exactly zero, no matter how many steps it takes or how the car oscillates in between, since the intermediate terms telescope away


 Yes — here's exactly how a size gets computed, so you can verify it against what you're seeing:

Offline (export_sensor_weights.py), for sensor i sensing tile type T: value = ||W_first_layer[:, i's 5 columns] · normalized_onehot(T)||₂ — the L2 norm of that sensor's contribution to all 128 first-hidden-layer units, given it's reading T right now. That's the raw number sitting in the JSON's values array.
Live (SensorWeightOverlay.Update()): looks up that sensor's value for its current reading, normalizes it against max_value — the single largest entry across the whole table (all 24 sensors × all 5 possible tile types = 120 numbers) — giving a 0–1 fraction t.
diameter = lerp(minDotRadius=3, maxDotRadius=22, t) * 2.
For your loaded checkpoint, max_value = 6.4600. So a dot at max size means that (sensor, currently-sensed-tile) combination is at or near the single strongest first-layer pull anywhere in this network — not an absolute physical unit, just relative to this checkpoint's own biggest entry. Two things worth knowing when judging "does this look right":

Sizes are relative to this one checkpoint's table, not comparable across different checkpoints — a "big" dot in one export isn't necessarily the same raw value as a "big" dot in another.
If several sensors look similarly sized, that's plausible, not necessarily a bug — many forward sensors are likely reading the same tile type (Asphalt) simultaneously, so they're all looking up the same handful of table entries.
To actually verify a specific dot rather than eyeball it: open Assets/CarAgent.sensor_weights.json — it's flat, values[sensor_index * 5 + tile_type_index] (tile_type order: Slippery=0, SpeedLimited=1, Terminal=2, Asphalt=3, Gravel=4). Pick a sensor, note what color it's showing (= what tile it's reading), look up that entry, and check it's proportionally where you'd expect relative to max_value=6.46.

If most values cluster tightly and the dots end up hard to tell apart visually, I can switch the size mapping from linear to something like sqrt(t) to spread out the low end — want me to add that, or does the current spread look fine?
