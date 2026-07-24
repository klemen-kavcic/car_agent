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
| 4 | Current steering angle, normalized | 1 | `currentSteerAngle / maxSteerAngle` → roughly [-1, 1] |
| 5 | Reverse gear flag | 1 | `carController.reverseGear ? 1 : 0` |
| 6 | Brake-pedal-selected flag | 1 | `pedalIsBrake ? 1 : 0` |
| 7 | Pedal magnitude | 1 | `gasInput + brakeInput` (only one is ever nonzero) |
| 8 | Current tile type, one-hot | 4 | Normal / Slippery / SpeedLimited / Terminal |
| **Base subtotal** | | **16** | |
| 9 | Grid sensor readings, one-hot per point | 37 × 4 = 148 | `CarSensor` — stadium-shaped pattern, radius 3, sensor spacing 3m |
| **Total** | | **164** | |


    - higher penalty later after it is more successful
    - different sensor placement to make them further out
    - checked it uses ADAM

    - remove brakeInput as condition for reverse
    - fixed clamping for pedal press
        - mlagents cover the raw outputs from neural netwroks, nn outputs a value and a distribuition, then magents choose a value and scale it to -1 - 1, so i scaled and then clamped the pedal press to make it 0-1, before there was a bug; ContinuousActions[1] is a raw Gaussian sample clipped by ML-Agents to [-1, 1]
    - training cirriculum for the first few steps
    

    Referencing the hpc folder without copying: Yes, this is possible — Unity's Project window only ever shows Assets/, but you can create a symlink/junction inside Assets/ that points at an external folder, and Unity will follow it and show the contents as if they were really there, with no physical duplication on disk. On Windows:


    mklink /J "Assets\Models\HpcResults" "D:\path\to\your\local\results\folder"
