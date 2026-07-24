using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using UnityEngine.InputSystem;

public class CarAgent : Agent
{
    public car_component carController;
    public Transform goal;
    public Transform spawnPoint;
    public Rigidbody rb;

    public Vector2 spawnAreaMin = new Vector2(-20f, -20f);
    public Vector2 spawnAreaMax = new Vector2(20f, 20f);
    public Vector2 goalAreaMin = new Vector2(-20f, -20f);
    public Vector2 goalAreaMax = new Vector2(20f, 20f);

    [Tooltip("Forces the bootstrap curriculum off, regardless of the bootstrap_curriculum_enabled " +
        "environment parameter. Environment parameters only reach the Academy when mlagents-learn is " +
        "attached, so this lets you toggle curriculum off from the Inspector while just watching in the " +
        "Editor (Heuristic or Inference, no trainer attached).")]
    public bool disableCurriculumInEditor = false;

    // Bootstrap curriculum: teaches basic controls before falling back to full-random goal
    // placement. See DetermineBootstrapStage() / PickBootstrapGoalPosition().
    private enum BootstrapStage { None, Front, Behind, Side }
    private BootstrapStage episodeBootstrapStage = BootstrapStage.None;
    [Tooltip("Which bootstrap curriculum stage placed this episode's goal - read-only display.")]
    public string currentBootstrapStageDisplay = "None";

    [Tooltip("Live cumulative reward for the episode in progress — read-only display, shows in Inspector during Play.")]
    public float currentEpisodeReward;
    [Tooltip("Cumulative reward from the most recently ended episode — read-only display.")]
    public float lastEpisodeReward;
    [Tooltip("How the most recent episode ended: Goal, Terminated, MaxStep, or Manual (N key / Force Next Episode) — read-only display.")]
    public string lastEpisodeEndReason = "";

    // Rolling window of recent genuine episode outcomes (Goal/Terminated/MaxStep — "Manual" skips
    // are excluded since they're a dev-testing artifact, not a real trial), used only to decide
    // when to escalate the terminal penalty (see terminal_penalty_escalation_* parameters below).
    // Per-agent-instance only: with --num-envs > 1 each parallel environment tracks its own
    // window and can escalate at a slightly different point, same caveat as TotalStepCount elsewhere.
    private Queue<bool> recentEpisodeGoals = new Queue<bool>();
    [Tooltip("Rolling goal-success rate over the last terminal_penalty_escalation_window episodes — read-only display.")]
    public float recentGoalRate;
    [Tooltip("Whether the terminal penalty has escalated to terminal_penalty_escalated_value yet — read-only display.")]
    public bool terminalPenaltyEscalated = false;

    // Set true exactly once whenever a genuine episode outcome (Goal/Terminated/MaxStep) gets
    // logged, and reset false at the start of each new episode. Also guards against MaxStep
    // firing twice for the same episode (OnActionReceived runs every Academy tick regardless of
    // DecisionPeriod, and the "StepCount >= MaxStep - 1" check is a >=, so without this guard it
    // fires on both the MaxStep-1 and MaxStep ticks before Agent's own internal auto-reset lands).
    // Starts true so the very first OnEpisodeBegin() call (no real "previous episode" yet) doesn't
    // spuriously report itself as untracked.
    private bool episodeOutcomeLogged = true;

    // Checkpoint progress reward: the spawn-to-goal distance is divided into CheckpointCount
    // equal bands. minDistanceThisEpisode only ever shrinks, so each band is awarded at most
    // once per episode, however many decisions it takes to reach it.
    private const int CheckpointCount = 100;
    private float spawnDistance;
    private float minDistanceThisEpisode;
    private int checkpointsAwarded;
    private Vector3 startPosition;
    public CarSensor carSensor;
    private int episodeStepCount = 0;
    // Which pedal the last action selected (mirrors carController.gasInput/brakeInput,
    // but stays meaningful even when pedalInput is 0, unlike the inputs themselves).
    private bool pedalIsBrake = false;

    // reward weights — set via environment_parameters in the YAML, defaults used if not provided
    private float rwGoal;
    private float rwApproachBonus;
    private float rwTimePenalty;
    private float rwCheckpoint;
    private float rwTerminalPenalty;
    private float rwTimePenaltyGrowth;

    void UpdateRewardWeights()
    {
        var ep = Academy.Instance.EnvironmentParameters;
        rwGoal                = ep.GetWithDefault("reward_goal",                 5.0f);
        rwApproachBonus       = ep.GetWithDefault("reward_approach_bonus",       3.0f);
        rwTimePenaltyGrowth   = ep.GetWithDefault("reward_time_penalty_growth",  0.0f);

        // Front/Behind bootstrap stages get their own (typically higher) approach bonus, to more
        // strongly reinforce driving straight toward/reversing straight into a goal that's directly
        // ahead/behind while basic controls are still being learned. Side stage and beyond (including
        // full-random placement once bootstrap ends) use the normal reward_approach_bonus above.
        if (episodeBootstrapStage == BootstrapStage.Front || episodeBootstrapStage == BootstrapStage.Behind)
            rwApproachBonus = ep.GetWithDefault("bootstrap_approach_bonus", rwApproachBonus);

        float timePenaltyBase      = ep.GetWithDefault("reward_time_penalty",       -0.01f);
        float checkpointRewardBase = ep.GetWithDefault("reward_checkpoint",          0.05f);
        float terminalPenaltyBase  = ep.GetWithDefault("reward_terminal_penalty",   -5.0f);

        // One-time step-up (not a gradual ramp) of the terminal penalty once the agent has been
        // consistently reaching the goal — tighten safety once it already drives competently,
        // rather than punishing crashes this harshly from the very first, still-clumsy episodes.
        // Latches permanently once triggered: doesn't revert if the rate later dips back down.
        bool terminalPenaltyEscalationEnabled = ep.GetWithDefault("terminal_penalty_escalation_enabled", 0f) >= 0.5f;
        float terminalPenaltyEscalationGoalRate = ep.GetWithDefault("terminal_penalty_escalation_goal_rate", 0.75f);
        int terminalPenaltyEscalationWindow = Mathf.Max(1, (int)ep.GetWithDefault("terminal_penalty_escalation_window", 20f));
        float terminalPenaltyEscalatedValue = ep.GetWithDefault("terminal_penalty_escalated_value", -15.0f);

        recentGoalRate = recentEpisodeGoals.Count > 0 ? CountTrue(recentEpisodeGoals) / (float)recentEpisodeGoals.Count : 0f;
        if (terminalPenaltyEscalationEnabled && !terminalPenaltyEscalated
            && recentEpisodeGoals.Count >= terminalPenaltyEscalationWindow
            && recentGoalRate >= terminalPenaltyEscalationGoalRate)
        {
            terminalPenaltyEscalated = true;
            Debug.Log($"[CarAgent] Terminal penalty escalated to {terminalPenaltyEscalatedValue} at step " +
                $"{Academy.Instance.TotalStepCount:N0} (rolling goal rate {recentGoalRate:P0}).");
            // One-shot report per environment — StatsReporter on the trainer side aggregates these
            // across every parallel worker into a single running total, printed as "Escalated: x/N"
            // by the ConsoleWriter patch (stats_patched.py) once a matching Custom/NumEnvs is seen.
            Academy.Instance.StatsRecorder.Add("Custom/EscalatedEnvironments", 1.0f, StatAggregationMethod.Sum);
            Academy.Instance.StatsRecorder.Add("Custom/NumEnvs", ep.GetWithDefault("training_num_envs", 1f), StatAggregationMethod.MostRecent);
        }
        rwTerminalPenalty = terminalPenaltyEscalated ? terminalPenaltyEscalatedValue : terminalPenaltyBase;

        rwTimePenalty = timePenaltyBase;
        rwCheckpoint  = checkpointRewardBase;
    }

    static int CountTrue(Queue<bool> values)
    {
        int count = 0;
        foreach (bool v in values) if (v) count++;
        return count;
    }

    // Feeds the rolling window that terminal_penalty_escalation_* uses to decide when to
    // step up the terminal penalty. Called for every genuine episode outcome (Goal/Terminated/
    // MaxStep) — see EndEpisodeWithLog and the MaxStep branch of OnActionReceived.
    void RecordEpisodeOutcome(bool wasGoal)
    {
        int window = Mathf.Max(1, (int)Academy.Instance.EnvironmentParameters.GetWithDefault("terminal_penalty_escalation_window", 20f));
        recentEpisodeGoals.Enqueue(wasGoal);
        while (recentEpisodeGoals.Count > window)
            recentEpisodeGoals.Dequeue();
    }

    public override void Initialize()
    {
        startPosition = spawnPoint.position;
    }

    void Update()
    {
        currentEpisodeReward = GetCumulativeReward();

        Keyboard kb = Keyboard.current;
        if (kb != null && kb.qKey.wasPressedThisFrame)
            carController.ToggleReverseGear();
        if (kb != null && kb.nKey.wasPressedThisFrame)
            ForceNextEpisode();

        // Controller: East face button (Circle/B) toggles reverse gear, same as Q.
        Gamepad gp = Gamepad.current;
        if (gp != null && gp.buttonEast.wasPressedThisFrame)
            carController.ToggleReverseGear();
    }

    // Snapshots the episode's final reward/end-reason for the Inspector display, then ends it.
    void EndEpisodeWithLog(string reason)
    {
        lastEpisodeReward = GetCumulativeReward();
        lastEpisodeEndReason = reason;
        if (reason != "Manual")
            RecordEpisodeOutcome(reason == "Goal");
        EndEpisode();
    }

    // Dev/testing convenience: end the current episode early with no extra reward, so a
    // fresh spawn/goal pair rolls immediately. Callable via the N key or, when paused,
    // right-click the component header in the Inspector during Play mode.
    [ContextMenu("Force Next Episode")]
    public void ForceNextEpisode()
    {
        EndEpisodeWithLog("Manual");
    }

    // Bootstrap curriculum stage for the upcoming episode: for the first bootstrap_stage_*_steps
    // (global steps / training_num_envs, same convention as every other curriculum in this file),
    // the goal is placed via PickBootstrapGoalPosition() instead of full-random placement, and
    // all tiles are forced Normal (see OnEpisodeBegin). Sequential: Front -> Behind -> Side, then
    // None forever after (full-random placement, normal tile mix resumes).
    BootstrapStage DetermineBootstrapStage()
    {
        if (disableCurriculumInEditor) return BootstrapStage.None;

        var ep = Academy.Instance.EnvironmentParameters;
        if (ep.GetWithDefault("bootstrap_curriculum_enabled", 1f) < 0.5f) return BootstrapStage.None;

        float numEnvs = Mathf.Max(1f, ep.GetWithDefault("training_num_envs", 1f));
        float frontSteps  = ep.GetWithDefault("bootstrap_stage_front_steps", 20_000f) / numEnvs;
        float behindSteps = ep.GetWithDefault("bootstrap_stage_behind_steps", 20_000f) / numEnvs;
        float sideSteps   = ep.GetWithDefault("bootstrap_stage_side_steps", 20_000f) / numEnvs;

        float step = Academy.Instance.TotalStepCount;
        if (step < frontSteps) return BootstrapStage.Front;
        if (step < frontSteps + behindSteps) return BootstrapStage.Behind;
        if (step < frontSteps + behindSteps + sideSteps) return BootstrapStage.Side;
        return BootstrapStage.None;
    }

    public override void OnEpisodeBegin()
    {
        // Diagnostic: if the previous episode ended without going through Goal/Terminated/MaxStep,
        // it vanishes from the Episodes vs Goals+Terminated+MaxStep reconciliation (see
        // check_episode_stats.py) with no trace of why. This surfaces it explicitly instead.
        if (!episodeOutcomeLogged)
        {
            Debug.LogWarning("[CarAgent] Previous episode ended without a tracked outcome (Goal/Terminated/MaxStep).");
            Academy.Instance.StatsRecorder.Add("Custom/UntrackedEpisodeEnd", 1.0f, StatAggregationMethod.Sum);
        }
        episodeOutcomeLogged = false;

        rb.linearVelocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;
        episodeStepCount = 0;

        episodeBootstrapStage = DetermineBootstrapStage();
        currentBootstrapStageDisplay = episodeBootstrapStage.ToString();

        UpdateRewardWeights();

        if (carController.gridManager != null)
        {
            carController.gridManager.forceAllNormal = episodeBootstrapStage != BootstrapStage.None;
            carController.gridManager.Regenerate();
        }

        Vector3 spawnPos;
        int attempts = 0;
        do {
            float spawnX = Random.Range(spawnAreaMin.x, spawnAreaMax.x);
            float spawnZ = Random.Range(spawnAreaMin.y, spawnAreaMax.y);
            spawnPos = new Vector3(spawnX, startPosition.y, spawnZ);
        } while (!IsNormalTile(spawnPos) && ++attempts < 100);
        transform.position = spawnPos;
        transform.rotation = Quaternion.Euler(0f, Random.Range(0f, 360f), 0f);

        // Behind stage: sometimes spawn already in reverse gear, so the policy gets direct
        // experience of what reverse does (gas -> backward motion -> reward, since the goal is
        // behind it) without first having to discover the gear-toggle action through random
        // exploration. Safe to set directly - CanShiftGear() only requires low speed, which spawn
        // (zero velocity) always satisfies. The policy's own discrete gear action can still flip
        // it back on the very first step; this just raises how often reverse gets tried at all.
        if (episodeBootstrapStage == BootstrapStage.Behind)
        {
            float spawnReverseProb = Academy.Instance.EnvironmentParameters.GetWithDefault(
                "bootstrap_behind_spawn_reverse_prob", 0.5f);
            carController.reverseGear = Random.value < spawnReverseProb;
        }

        goal.position = PickGoalPosition(spawnPos);

        spawnDistance = Vector3.Distance(transform.position, goal.position);
        minDistanceThisEpisode = spawnDistance;
        checkpointsAwarded = 0;
    }

    bool IsNormalTile(Vector3 worldPos)
    {
        if (carController.gridManager == null) return true;
        return carController.gridManager.GetTileAt(worldPos) == TileType.Normal;
    }

    // Bootstrap curriculum takes placement priority while active; once it's finished (or
    // disabled), goal placement is pure full-random across the whole goal area.
    Vector3 PickGoalPosition(Vector3 spawnPos)
    {
        if (episodeBootstrapStage != BootstrapStage.None)
            return PickBootstrapGoalPosition(spawnPos);

        Vector3 goalPos;
        int attempts = 0;
        do
        {
            float goalX = Random.Range(goalAreaMin.x, goalAreaMax.x);
            float goalZ = Random.Range(goalAreaMin.y, goalAreaMax.y);
            goalPos = new Vector3(goalX, goal.position.y, goalZ);
        } while (!IsNormalTile(goalPos) && ++attempts < 100);
        return goalPos;
    }

    // Front: goal straight ahead (basic throttle+steering). Behind: goal straight behind (forces
    // reverse gear - see the flipped approach-bonus dot in OnTriggerEnter). Side: randomised
    // angle/distance for variety, bridging toward the eventual full-random placement.
    Vector3 PickBootstrapGoalPosition(Vector3 spawnPos)
    {
        var ep = Academy.Instance.EnvironmentParameters;
        Vector3 goalPos;
        int attempts = 0;

        switch (episodeBootstrapStage)
        {
            case BootstrapStage.Front:
            {
                float distance = ep.GetWithDefault("bootstrap_front_distance", 6f);
                do
                {
                    goalPos = spawnPos + transform.forward * distance;
                    goalPos.x = Mathf.Clamp(goalPos.x, goalAreaMin.x, goalAreaMax.x);
                    goalPos.y = goal.position.y;
                    goalPos.z = Mathf.Clamp(goalPos.z, goalAreaMin.y, goalAreaMax.y);
                } while (!IsNormalTile(goalPos) && ++attempts < 100);
                return goalPos;
            }

            case BootstrapStage.Behind:
            {
                float distance = ep.GetWithDefault("bootstrap_behind_distance", 6f);
                do
                {
                    goalPos = spawnPos - transform.forward * distance;
                    goalPos.x = Mathf.Clamp(goalPos.x, goalAreaMin.x, goalAreaMax.x);
                    goalPos.y = goal.position.y;
                    goalPos.z = Mathf.Clamp(goalPos.z, goalAreaMin.y, goalAreaMax.y);
                } while (!IsNormalTile(goalPos) && ++attempts < 100);
                return goalPos;
            }

            default: // BootstrapStage.Side
            {
                float minDist  = ep.GetWithDefault("bootstrap_side_distance_min", 8f);
                float maxDist  = ep.GetWithDefault("bootstrap_side_distance_max", 15f);
                float minAngle = ep.GetWithDefault("bootstrap_side_angle_min_deg", 30f);
                float maxAngle = ep.GetWithDefault("bootstrap_side_angle_max_deg", 150f);
                do
                {
                    float side = Random.value < 0.5f ? -1f : 1f;                 // left/right
                    float angle = side * Random.Range(minAngle, maxAngle);       // deg from forward
                    float distance = Random.Range(minDist, Mathf.Max(minDist, maxDist));
                    Vector3 dir = Quaternion.AngleAxis(angle, Vector3.up) * transform.forward;
                    goalPos = spawnPos + dir * distance;
                    goalPos.x = Mathf.Clamp(goalPos.x, goalAreaMin.x, goalAreaMax.x);
                    goalPos.y = goal.position.y;
                    goalPos.z = Mathf.Clamp(goalPos.z, goalAreaMin.y, goalAreaMax.y);
                } while (!IsNormalTile(goalPos) && ++attempts < 100);
                return goalPos;
            }
        }
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        carSensor?.UpdateReadings();

        Vector3 toGoal = goal.position - transform.position;
        sensor.AddObservation(transform.InverseTransformDirection(toGoal.normalized)); // 3
        sensor.AddObservation(toGoal.magnitude);                                        // 1
        sensor.AddObservation(transform.InverseTransformDirection(rb.linearVelocity)); // 3
        sensor.AddObservation(transform.InverseTransformDirection(rb.angularVelocity).y); // 1 - yaw rate
        sensor.AddObservation(carController.currentSteerAngle / carController.maxSteerAngle); // 1
        sensor.AddObservation(carController.reverseGear ? 1f : 0f); // 1
        sensor.AddObservation(pedalIsBrake ? 1f : 0f); // 1 - which pedal is selected
        sensor.AddObservation(carController.gasInput + carController.brakeInput); // 1 - pedal magnitude (only one is ever nonzero)

        // Current tile type, one-hot (4 values)                                        // 4
        AddTileOneHot(sensor, carController.currentTileType);

        // sensor readings: carSensor.SensorCount × 4 values (one-hot per point)
        // If the sensor shape/radius/ring settings change, update Space Size to: 16 + SensorCount * 4
        if (carSensor != null && carSensor.readings != null)
        {
            foreach (var t in carSensor.readings)
                AddTileOneHot(sensor, t);
        }
        else
        {
            // fallback: 13 zeros per reading × 4 (one-hot) for default circular radius=2
            int count = carSensor != null ? carSensor.SensorCount : 13;
            for (int i = 0; i < count * 4; i++)
                sensor.AddObservation(0f);
        }
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        float steer = actions.ContinuousActions[0];
        float pedal = (actions.ContinuousActions[1] + 1f) / 2f;
        pedal = Mathf.Clamp01(pedal); // safety net only, for rare slight overshoot
        pedalIsBrake = actions.DiscreteActions[1] == 1;

        carController.steerDeltaInput = Mathf.Clamp(steer, -1f, 1f);  //clamping just for the safety net, to eliminate any float rounding errors
        
        
        // Mutually exclusive by construction: only the selected pedal ever gets a nonzero input,
        // so the agent can no longer press gas and brake at once (car_component previously allowed it).
        carController.gasInput = pedalIsBrake ? 0f : pedal;
        carController.brakeInput = pedalIsBrake ? pedal : 0f;
        carController.RequestReverseGear(actions.DiscreteActions[0] == 1);

        episodeStepCount++;
        float scaledTimePenalty = rwTimePenalty * (1f + episodeStepCount * rwTimePenaltyGrowth);
        AddReward(scaledTimePenalty);

        float currentDistance = Vector3.Distance(transform.position, goal.position);
        if (currentDistance < minDistanceThisEpisode)
        {
            minDistanceThisEpisode = currentDistance;
            if (spawnDistance > 0f)
            {
                int checkpointsReached = Mathf.Clamp(
                    Mathf.FloorToInt((spawnDistance - minDistanceThisEpisode) / spawnDistance * CheckpointCount),
                    0, CheckpointCount);
                if (checkpointsReached > checkpointsAwarded)
                {
                    AddReward((checkpointsReached - checkpointsAwarded) * rwCheckpoint);
                    checkpointsAwarded = checkpointsReached;
                }
            }
        }

        if (carController.currentTileType == TileType.Terminal)
        {
            AddReward(rwTerminalPenalty);
            Academy.Instance.StatsRecorder.Add("Custom/Terminated", 1.0f, StatAggregationMethod.Sum);
            episodeOutcomeLogged = true;
            EndEpisodeWithLog("Terminated");
            return;
        }

        if (transform.position.y < -2f || Vector3.Dot(transform.up, Vector3.up) < 0f)
        {
            AddReward(rwTerminalPenalty);
            Academy.Instance.StatsRecorder.Add("Custom/Terminated", 1.0f, StatAggregationMethod.Sum);
            episodeOutcomeLogged = true;
            EndEpisodeWithLog("Terminated");
            return;
        }

        // Guarded by !episodeOutcomeLogged: OnActionReceived runs every Academy tick regardless of
        // DecisionPeriod (TakeActionsBetweenDecisions defaults true), and this check is a >= not an
        // ==, so without the guard it fires on both the MaxStep-1 and MaxStep ticks - double-logging
        // one real episode end, before Agent's own internal auto-reset (Agent.cs AgentStep) lands.
        if (MaxStep > 0 && StepCount >= MaxStep - 1 && !episodeOutcomeLogged)
        {
            Academy.Instance.StatsRecorder.Add("Custom/MaxStepReached", 1.0f, StatAggregationMethod.Sum);
            episodeOutcomeLogged = true;
            // The Academy ends this episode automatically right after this step (no explicit
            // EndEpisode() call here), so snapshot the display fields directly instead of
            // going through EndEpisodeWithLog.
            lastEpisodeReward = GetCumulativeReward();
            lastEpisodeEndReason = "MaxStep";
            RecordEpisodeOutcome(false);
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActions = actionsOut.ContinuousActions;
        var discreteActions = actionsOut.DiscreteActions;
        Keyboard kb = Keyboard.current;
        Gamepad gp = Gamepad.current;

        float horizontal = 0f;
        float gas = 0f;
        float brake = 0f;

        if (kb != null)
        {
            if (kb.aKey.isPressed) horizontal -= 1f;
            if (kb.dKey.isPressed) horizontal += 1f;
            if (kb.wKey.isPressed) gas = 1f;
            if (kb.sKey.isPressed) brake = 1f;
        }

        // Controller: left stick + analog triggers give true continuous values,
        // unlike the keyboard's hard 0/1 steps — combined with keyboard via Max/add so either device works.
        if (gp != null)
        {
            horizontal += gp.leftStick.x.ReadValue();
            gas = Mathf.Max(gas, gp.rightTrigger.ReadValue());
            brake = Mathf.Max(brake, gp.leftTrigger.ReadValue());
        }

        // The action space only allows one pedal at a time; brake wins if both are held
        // (e.g. W+S together), since braking is the safer default to fall back to.
        bool brakePedal = brake > 0f;
        float pedal = brakePedal ? brake : gas;

        continuousActions[0] = Mathf.Clamp(horizontal, -1f, 1f);
        continuousActions[1] = Mathf.Clamp01(pedal);

        // Echo the current gear (driven by the Q-toggle in Update()) rather than requesting
        // a fresh value here — otherwise RequestReverseGear() in OnActionReceived would fight
        // the toggle every physics step.
        discreteActions[0] = carController.reverseGear ? 1 : 0;
        discreteActions[1] = brakePedal ? 1 : 0;
    }

    void OnTriggerEnter(Collider other)
    {
        if (other.transform == goal)
        {
            AddReward(rwGoal);

            // bonus up to rwApproachBonus for approaching straight on (dot=1 = perfect, dot=0 = sideways).
            // Behind-stage episodes want the agent to reinforce reversing into the goal, not
            // spinning 180 degrees to face it forward - reuse the same rwApproachBonus channel,
            // just flip which facing direction counts as "approaching well."
            Vector3 approachForward = episodeBootstrapStage == BootstrapStage.Behind
                ? -transform.forward
                : transform.forward;
            float approachDot = Vector3.Dot(approachForward, rb.linearVelocity.normalized);
            AddReward(Mathf.Max(0f, approachDot) * rwApproachBonus);

            Academy.Instance.StatsRecorder.Add("Custom/GoalReached", 1.0f, StatAggregationMethod.Sum);
            episodeOutcomeLogged = true;
            EndEpisodeWithLog("Goal");
        }
    }

    void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.CompareTag("Wall"))
        {
            AddReward(rwTerminalPenalty);
            Academy.Instance.StatsRecorder.Add("Custom/Terminated", 1.0f, StatAggregationMethod.Sum);
            episodeOutcomeLogged = true;
            EndEpisodeWithLog("Terminated");
        }
    }

    void AddTileOneHot(VectorSensor sensor, TileType type)
    {
        sensor.AddObservation(type == TileType.Normal       ? 1f : 0f);
        sensor.AddObservation(type == TileType.Slippery     ? 1f : 0f);
        sensor.AddObservation(type == TileType.SpeedLimited ? 1f : 0f);
        sensor.AddObservation(type == TileType.Terminal     ? 1f : 0f);
    }
}
