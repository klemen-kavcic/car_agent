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

    [Tooltip("When true, skips the step-based bootstrap calculation entirely and always uses " +
        "editorForcedStage instead - lets you jump straight to testing a specific stage (e.g. " +
        "Behind, with a trained model in Inference) without waiting for enough steps to actually " +
        "elapse in the Editor. Takes priority over disableCurriculumInEditor if both are set. " +
        "Leave false for real training runs - environment parameters/TotalStepCount take over " +
        "once mlagents-learn is attached regardless, but there's no reason to leave it on.")]
    public bool forceStageInEditor = false;
    [Tooltip("Which stage DetermineBootstrapStage() returns when forceStageInEditor is true.")]
    public BootstrapStage editorForcedStage = BootstrapStage.Behind;

    // Bootstrap curriculum: teaches basic controls before falling back to full-random goal
    // placement. See DetermineBootstrapStage() / PickBootstrapGoalPosition().
    public enum BootstrapStage { None, Front, Behind, DeadEnd, Side }
    private BootstrapStage episodeBootstrapStage = BootstrapStage.None;
    [Tooltip("Which bootstrap curriculum stage placed this episode's goal - read-only display.")]
    public string currentBootstrapStageDisplay = "None";

    // Tracks the previous episode's stage so a genuine transition (Front->Behind->Side->None)
    // can be reported exactly once per environment, the moment it happens - see OnEpisodeBegin.
    // Starts equal to episodeBootstrapStage's own default (None) specifically so the very first
    // episode of a run (which is never a "transition") doesn't spuriously report one; guarded
    // additionally by hasBootstrapStageHistory since a curriculum-disabled run also stays at
    // None forever and must never report a false "transition into None".
    private BootstrapStage previousEpisodeBootstrapStage = BootstrapStage.None;
    private bool hasBootstrapStageHistory = false;

    [Tooltip("Live cumulative reward for the episode in progress — read-only display, shows in Inspector during Play.")]
    public float currentEpisodeReward;
    [Tooltip("Cumulative reward from the most recently ended episode — read-only display.")]
    public float lastEpisodeReward;
    [Tooltip("How the most recent episode ended: Goal, Terminated, MaxStep, or Manual (N key / Force Next Episode) — read-only display.")]
    public string lastEpisodeEndReason = "";

    [Tooltip("Whether the terminal penalty has escalated to terminal_penalty_escalated_value yet — read-only display.")]
    public bool terminalPenaltyEscalated = false;

    [Tooltip("Whether the goal reward has escalated to goal_reward_escalated_value yet — read-only display.")]
    public bool goalRewardEscalated = false;

    // Set true exactly once whenever a genuine episode outcome (Goal/Terminated/MaxStep) gets
    // logged, and reset false at the start of each new episode. Also guards against MaxStep
    // firing twice for the same episode (OnActionReceived runs every Academy tick regardless of
    // DecisionPeriod, and the "StepCount >= MaxStep - 1" check is a >=, so without this guard it
    // fires on both the MaxStep-1 and MaxStep ticks before Agent's own internal auto-reset lands).
    // Starts true so the very first OnEpisodeBegin() call (no real "previous episode" yet) doesn't
    // spuriously report itself as untracked.
    private bool episodeOutcomeLogged = true;

    // Guards OnTriggerEnter against firing more than once for the same goal arrival. If the car's
    // Rigidbody hierarchy has more than one non-wheel Collider (e.g. separate chassis/body-panel
    // colliders), Unity calls OnTriggerEnter once per colliding collider that overlaps the goal's
    // trigger volume in the same physics step - without this guard, that means AddReward(rwGoal)
    // and EndEpisode() could both fire twice for one genuine goal touch, silently doubling the
    // reward. Reset false at the start of each episode, same convention as episodeOutcomeLogged.
    private bool goalTriggeredThisEpisode = false;

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
    // Which pedal the last action selected (mirrors carController.accelerationInput/brakeInput,
    // but stays meaningful even when pedalInput is 0, unlike the inputs themselves).
    private bool pedalIsBrake = false;

    // reward weights — set via environment_parameters in the YAML, defaults used if not provided
    private float rwGoal;
    private float rwApproachBonus;
    private float rwTimePenalty;
    private float rwCheckpoint;
    private float rwTerminalPenalty;
    private float rwTimePenaltyGrowth;

    // Idle penalty: escalating cost for standing still too long, on top of the flat time
    // penalty. See UpdateRewardWeights() / OnActionReceived() for why this needs its own
    // threshold/grace/growth rather than reusing reward_time_penalty_growth.
    private bool idlePenaltyEnabled;
    private float idleSpeedThreshold;
    private int idleGraceSteps;
    private float rwIdlePenaltyBase;
    private float rwIdlePenaltyGrowth;
    private int idleStepCounter = 0;

    void UpdateRewardWeights()
    {
        var ep = Academy.Instance.EnvironmentParameters;
        float goalRewardBase = ep.GetWithDefault("reward_goal",                 5.0f);
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

        // One-time step-up (not a gradual ramp) of the goal reward once training has run long
        // enough — same mechanism/rationale as the terminal-penalty escalation right below (step-
        // based so every parallel worker escalates at essentially the same point, latches
        // permanently once triggered). Lets a run start with a smaller goal reward (e.g. to keep
        // early gradients from being dominated by it while the policy is still mostly learning
        // basic control from the approach/checkpoint rewards) and step up once the agent is
        // already reasonably competent.
        bool goalRewardEscalationEnabled = ep.GetWithDefault("goal_reward_escalation_enabled", 0f) >= 0.5f;
        float goalRewardEscalatedValue = ep.GetWithDefault("goal_reward_escalated_value", 5.0f);
        if (goalRewardEscalationEnabled && !goalRewardEscalated)
        {
            float numEnvs = Mathf.Max(1f, ep.GetWithDefault("training_num_envs", 1f));
            float decisionPeriod = decisionRequester != null ? Mathf.Max(1, decisionRequester.DecisionPeriod) : 1f;
            float escalationStep = ep.GetWithDefault("goal_reward_escalation_step", 10_000_000f) * decisionPeriod / numEnvs;
            if (Academy.Instance.TotalStepCount >= escalationStep)
            {
                goalRewardEscalated = true;
                Debug.Log($"[CarAgent] Goal reward escalated to {goalRewardEscalatedValue} at step " +
                    $"{Academy.Instance.TotalStepCount:N0}.");
                // Separate stat name from Custom/EscalatedEnvironments (terminal penalty) so the two
                // escalations can be tracked/printed independently — see stats_patched.py.
                Academy.Instance.StatsRecorder.Add("Custom/EscalatedGoalRewardEnvironments", 1.0f, StatAggregationMethod.Sum);
                Academy.Instance.StatsRecorder.Add("Custom/NumEnvs", numEnvs, StatAggregationMethod.MostRecent);
            }
        }
        rwGoal = goalRewardEscalated ? goalRewardEscalatedValue : goalRewardBase;

        // One-time step-up (not a gradual ramp) of the terminal penalty once training has run
        // long enough — tighten safety once the agent should already drive competently, rather
        // than punishing crashes this harshly from the very first, still-clumsy episodes.
        // Step-based (not goal-rate-based) specifically so every parallel worker escalates at
        // essentially the same point: Academy.Instance.TotalStepCount is per-process, but all
        // workers are paced by the same DecisionPeriod so it stays in close lockstep across them -
        // unlike a per-worker rolling goal-rate window, which could previously drift by millions
        // of steps between the luckiest and unluckiest worker before each independently crossed
        // its own threshold. Same "global steps / training_num_envs" convention as
        // DetermineBootstrapStage(), corrected for DecisionPeriod the same way. Latches
        // permanently once triggered (monotonic anyway, since TotalStepCount only increases).
        bool terminalPenaltyEscalationEnabled = ep.GetWithDefault("terminal_penalty_escalation_enabled", 0f) >= 0.5f;
        float terminalPenaltyEscalatedValue = ep.GetWithDefault("terminal_penalty_escalated_value", -15.0f);
        if (terminalPenaltyEscalationEnabled && !terminalPenaltyEscalated)
        {
            float numEnvs = Mathf.Max(1f, ep.GetWithDefault("training_num_envs", 1f));
            float decisionPeriod = decisionRequester != null ? Mathf.Max(1, decisionRequester.DecisionPeriod) : 1f;
            float escalationStep = ep.GetWithDefault("terminal_penalty_escalation_step", 10_000_000f) * decisionPeriod / numEnvs;
            if (Academy.Instance.TotalStepCount >= escalationStep)
            {
                terminalPenaltyEscalated = true;
                Debug.Log($"[CarAgent] Terminal penalty escalated to {terminalPenaltyEscalatedValue} at step " +
                    $"{Academy.Instance.TotalStepCount:N0}.");
                // One-shot report per environment — StatsReporter on the trainer side aggregates these
                // across every parallel worker into a single running total, printed as "Escalated: x/N"
                // by the ConsoleWriter patch (stats_patched.py) once a matching Custom/NumEnvs is seen.
                Academy.Instance.StatsRecorder.Add("Custom/EscalatedEnvironments", 1.0f, StatAggregationMethod.Sum);
                Academy.Instance.StatsRecorder.Add("Custom/NumEnvs", numEnvs, StatAggregationMethod.MostRecent);
            }
        }
        rwTerminalPenalty = terminalPenaltyEscalated ? terminalPenaltyEscalatedValue : terminalPenaltyBase;

        rwTimePenalty = timePenaltyBase;
        rwCheckpoint  = checkpointRewardBase;

        // Idle penalty: escalating cost for standing still, layered on top of the flat time
        // penalty above. idle_speed_threshold defaults well below car_component's
        // reverseEngageSpeed (0.5 m/s) so a normal brake-to-shift-gear moment never trips it;
        // idle_grace_steps absorbs a brief full stop before any penalty starts accruing.
        // Growth (not a flat idle penalty) matters specifically because a flat cost can just be
        // "priced in" and tolerated for the rest of the episode - escalating it means standing
        // still eventually costs strictly more than the bounded risk of attempting to reverse
        // out of a dead end, instead of competing with it forever at a fixed rate.
        idlePenaltyEnabled = ep.GetWithDefault("idle_penalty_enabled", 0f) >= 0.5f;
        idleSpeedThreshold = ep.GetWithDefault("idle_speed_threshold", 0.001f);
        idleGraceSteps = Mathf.RoundToInt(ep.GetWithDefault("idle_grace_steps", 20f));
        rwIdlePenaltyBase = ep.GetWithDefault("idle_penalty_base", -0.02f);
        rwIdlePenaltyGrowth = ep.GetWithDefault("idle_penalty_growth", 0.05f);
    }

    private DecisionRequester decisionRequester;

    public override void Initialize()
    {
        startPosition = spawnPoint.position;
        decisionRequester = GetComponent<DecisionRequester>();
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

    // MaxStep detection lives here, NOT in OnActionReceived, specifically because this runs every
    // physics tick unconditionally - OnActionReceived only runs on decision ticks when
    // TakeActionsBetweenDecisions is off (confirmed off in this project's DecisionRequester), and
    // those are gated on the GLOBAL Academy step count, not this episode's own StepCount. Since
    // episodes can end (Goal/Terminated) at any arbitrary tick, a new episode's start is
    // essentially randomly phase-shifted relative to the decision-tick schedule - so roughly 4
    // times out of 5 (DecisionPeriod=5), the exact tick where StepCount reaches MaxStep would NOT
    // be a decision tick, and OnActionReceived-based detection would silently miss it entirely,
    // even though Agent's own internal MaxStep check (Agent.cs AgentStep, subscribed to
    // Academy.AgentAct, which fires every tick regardless of decision period) still ends the
    // episode right on schedule - this was the actual cause of the "untracked episode" gap.
    // Guarded by !episodeOutcomeLogged since this is a >= check, not ==, so it would otherwise
    // fire on both the MaxStep-1 and MaxStep ticks before Agent's internal auto-reset lands.
    void FixedUpdate()
    {
        if (MaxStep > 0 && StepCount >= MaxStep - 1 && !episodeOutcomeLogged)
        {
            Academy.Instance.StatsRecorder.Add("Custom/MaxStepReached", 1.0f, StatAggregationMethod.Sum);
            episodeOutcomeLogged = true;
            // The Academy ends this episode automatically right after this step (no explicit
            // EndEpisode() call here), so snapshot the display fields directly instead of
            // going through EndEpisodeWithLog.
            lastEpisodeReward = GetCumulativeReward();
            lastEpisodeEndReason = "MaxStep";
        }
    }

    // Snapshots the episode's final reward/end-reason for the Inspector display, then ends it.
    void EndEpisodeWithLog(string reason)
    {
        lastEpisodeReward = GetCumulativeReward();
        lastEpisodeEndReason = reason;
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

    // Start/end of the Behind/DeadEnd stages in the same tick units as Academy.TotalStepCount,
    // cached by DetermineBootstrapStage() each episode so OnEpisodeBegin can compute how far
    // through the stage the current episode falls (used to decay each stage's own
    // bootstrap_*_spawn_reverse_prob_start/_end), without re-reading/recomputing the same
    // environment parameters a second time.
    private float behindStageStartStep;
    private float behindStageEndStep;
    private float deadEndStageStartStep;
    private float deadEndStageEndStep;

    // Bootstrap curriculum stage for the upcoming episode: for the first bootstrap_stage_*_steps
    // (global steps / training_num_envs, same convention as every other curriculum in this file),
    // the goal is placed via PickBootstrapGoalPosition() instead of full-random placement, and
    // all tiles are forced Normal (see OnEpisodeBegin). Sequential: Front -> Behind -> DeadEnd ->
    // Side, then None forever after (full-random placement, normal tile mix resumes).
    BootstrapStage DetermineBootstrapStage()
    {
        if (forceStageInEditor) return editorForcedStage;
        if (disableCurriculumInEditor) return BootstrapStage.None;

        var ep = Academy.Instance.EnvironmentParameters;
        if (ep.GetWithDefault("bootstrap_curriculum_enabled", 1f) < 0.5f) return BootstrapStage.None;

        float numEnvs = Mathf.Max(1f, ep.GetWithDefault("training_num_envs", 1f));

        // Academy.TotalStepCount (used below) ticks every environment step regardless of
        // DecisionPeriod, but bootstrap_stage_*_steps are GLOBAL DECISION steps - matching the
        // trainer's own Step:/max_steps convention (decisions summed across workers). Multiplying
        // by DecisionPeriod converts back into the tick units TotalStepCount actually counts in,
        // so e.g. bootstrap_stage_front_steps: 200000 with --num-envs 32 really means the Front
        // stage lasts until the trainer's Step: counter reaches 200000, not 200000/DecisionPeriod.
        float decisionPeriod = decisionRequester != null ? Mathf.Max(1, decisionRequester.DecisionPeriod) : 1f;
        float frontSteps   = ep.GetWithDefault("bootstrap_stage_front_steps", 20_000f) * decisionPeriod / numEnvs;
        float behindSteps  = ep.GetWithDefault("bootstrap_stage_behind_steps", 20_000f) * decisionPeriod / numEnvs;
        float deadEndSteps = ep.GetWithDefault("bootstrap_stage_deadend_steps", 20_000f) * decisionPeriod / numEnvs;
        float sideSteps    = ep.GetWithDefault("bootstrap_stage_side_steps", 20_000f) * decisionPeriod / numEnvs;

        behindStageStartStep = frontSteps;
        behindStageEndStep = frontSteps + behindSteps;
        deadEndStageStartStep = behindStageEndStep;
        deadEndStageEndStep = behindStageEndStep + deadEndSteps;

        float step = Academy.Instance.TotalStepCount;
        if (step < frontSteps) return BootstrapStage.Front;
        if (step < behindStageEndStep) return BootstrapStage.Behind;
        if (step < deadEndStageEndStep) return BootstrapStage.DeadEnd;
        if (step < deadEndStageEndStep + sideSteps) return BootstrapStage.Side;
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
        goalTriggeredThisEpisode = false;

        rb.linearVelocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;
        episodeStepCount = 0;
        idleStepCounter = 0;

        // Neither of these is reset by anything else - car_component has no episode concept of
        // its own, so without this, wheels/gear silently carry over whatever they were at the
        // instant the previous episode ended (e.g. still turned hard from a spin, or stuck in
        // reverse from a Behind-stage episode). The Behind-stage branch below can still put the
        // agent back into reverse on purpose; this just makes "forward, wheels straight" the
        // default starting point for every episode instead of whatever state was left behind.
        carController.currentSteerAngle = 0f;
        carController.reverseGear = false;
        carController.accelerationInput = 0f;
        carController.brakeInput = 0f;
        carController.ResetSlipperyDisturbance();

        episodeBootstrapStage = DetermineBootstrapStage();
        currentBootstrapStageDisplay = episodeBootstrapStage.ToString();

        // Report the exact step of each stage transition once per environment (Sum-aggregated,
        // same one-shot-per-env pattern as the terminal-penalty/goal-reward escalation stats
        // above), so it's visible directly in the trainer log/TensorBoard instead of having to
        // infer it from an inflection in the reward curve - see ConsoleWriter in stats_patched.py
        // for the matching "changed since last report" print logic.
        if (hasBootstrapStageHistory && episodeBootstrapStage != previousEpisodeBootstrapStage)
        {
            Debug.Log($"[CarAgent] Bootstrap stage changed: {previousEpisodeBootstrapStage} -> " +
                $"{episodeBootstrapStage} at step {Academy.Instance.TotalStepCount:N0}.");
            Academy.Instance.StatsRecorder.Add(
                $"Custom/BootstrapStageEntered{episodeBootstrapStage}", 1.0f, StatAggregationMethod.Sum);
        }
        previousEpisodeBootstrapStage = episodeBootstrapStage;
        hasBootstrapStageHistory = true;

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

        // DeadEnd stage: snap the spawn heading to one of the 4 grid-aligned cardinal
        // directions instead of a continuous random yaw. The pocket built further below is
        // placed using exact cellSize multiples of transform.forward/right - that only lands
        // on clean, unbroken grid cells (no staggered/gapped diagonal wall) when forward/right
        // are themselves exactly axis-aligned.
        int deadEndDepthCells = 0;
        if (episodeBootstrapStage == BootstrapStage.DeadEnd)
        {
            float cardinalYaw = Random.Range(0, 4) * 90f;
            transform.rotation = Quaternion.Euler(0f, cardinalYaw, 0f);

            // Same reverse-gear-spawn trick as the Behind stage below, but decaying all the way
            // to 0 by the end of this stage instead of Behind's nonzero floor - DeadEnd is the
            // last bootstrap stage with anything to do with reversing, and Side/full-random
            // afterward offer no assistance at all, so by the time it ends the policy needs to
            // reliably decide to shift into reverse entirely on its own, from wall-sensor cues
            // rather than a goal conveniently placed behind it.
            var deadEndEp = Academy.Instance.EnvironmentParameters;
            float deadEndProgress = deadEndStageEndStep > deadEndStageStartStep
                ? Mathf.Clamp01((Academy.Instance.TotalStepCount - deadEndStageStartStep) / (deadEndStageEndStep - deadEndStageStartStep))
                : 0f;
            float deadEndSpawnReverseProbStart = deadEndEp.GetWithDefault("bootstrap_deadend_spawn_reverse_prob_start", 1.0f);
            float deadEndSpawnReverseProbEnd = deadEndEp.GetWithDefault("bootstrap_deadend_spawn_reverse_prob_end", 0.0f);
            carController.reverseGear = Random.value <
                Mathf.Lerp(deadEndSpawnReverseProbStart, deadEndSpawnReverseProbEnd, deadEndProgress);

            // Push the car deep into the pocket (near the closed end) instead of leaving it
            // right at the mouth - spawning at the entrance made escaping nearly free (the car
            // was already almost back out), which barely exercised the "recognize you're stuck
            // and reverse out" skill this stage exists to teach. spawnPos itself stays the
            // mouth/entrance reference for the goal-placement and wall-building math below;
            // only the car's actual starting transform is moved deeper.
            if (carController.gridManager != null)
            {
                deadEndDepthCells = Mathf.Max(1, Mathf.RoundToInt(
                    Academy.Instance.EnvironmentParameters.GetWithDefault("bootstrap_deadend_depth_cells", 3f)));
                float cellSize = carController.gridManager.cellSize;
                transform.position = spawnPos + transform.forward * ((deadEndDepthCells - 1) * cellSize);
            }
        }

        // Behind stage: sometimes spawn already in reverse gear, so the policy gets direct
        // experience of what reverse does (acceleration -> backward motion -> reward, since the goal is
        // behind it) without first having to discover the gear-toggle action through random
        // exploration. Safe to set directly - CanShiftGear() only requires low speed, which spawn
        // (zero velocity) always satisfies. The policy's own discrete gear action can still flip
        // it back on the very first step; this just raises how often reverse gets tried at all.
        // The probability itself decays linearly across the stage (bootstrap_behind_spawn_reverse_
        // prob_start -> _end): early on, near-certain reverse-spawn isolates "accelerate while
        // already in reverse" as the only thing being learned, with zero gear-decision confusion;
        // later, a lower probability forces the policy to also practice actually deciding to shift
        // into reverse itself, before bootstrap ends and it has to do that from full-random spawns.
        if (episodeBootstrapStage == BootstrapStage.Behind)
        {
            var ep = Academy.Instance.EnvironmentParameters;
            float behindProgress = behindStageEndStep > behindStageStartStep
                ? Mathf.Clamp01((Academy.Instance.TotalStepCount - behindStageStartStep) / (behindStageEndStep - behindStageStartStep))
                : 0f;
            float spawnReverseProbStart = ep.GetWithDefault("bootstrap_behind_spawn_reverse_prob_start", 1.0f);
            float spawnReverseProbEnd = ep.GetWithDefault("bootstrap_behind_spawn_reverse_prob_end", 0.3f);
            float spawnReverseProb = Mathf.Lerp(spawnReverseProbStart, spawnReverseProbEnd, behindProgress);
            carController.reverseGear = Random.value < spawnReverseProb;
        }

        goal.position = PickGoalPosition(spawnPos);

        // Behind stage: force a Terminal tile directly in front of the spawn point too, so driving
        // forward - instead of reversing toward the goal behind - ends the episode immediately.
        // A much stronger, unambiguous signal than the reverse-spawn-probability trick above,
        // which only ever nudges the odds of trying reverse rather than actively punishing
        // forward. Placed AFTER the goal so it can't affect the goal's own IsNormalTile retry loop.
        if (episodeBootstrapStage == BootstrapStage.Behind && carController.gridManager != null)
        {
            float terminalWallDistance = Academy.Instance.EnvironmentParameters.GetWithDefault(
                "bootstrap_behind_terminal_distance", 4f);
            Vector3 wallPos = spawnPos + transform.forward * terminalWallDistance;
            carController.gridManager.SetTileTypeAt(wallPos, TileType.Terminal);
        }

        // DeadEnd stage: enclose the (mouth-referenced) spawnPos in a narrow, one-cell-wide
        // pocket - Terminal walls flanking both sides of the lane the whole way, plus a Terminal
        // cap across the full width one cell past the end of it. forceAllNormal (set above)
        // already leaves the lane itself and the open area behind the mouth untouched (Normal),
        // so with the car now sitting deep inside this same corridor (see the position shift
        // above), the only way to reach the goal (placed behind the mouth - see
        // PickBootstrapGoalPosition) without ending the episode on a Terminal tile is to reverse
        // straight back out past the walled rows between it and the mouth. Unlike the single
        // Behind-stage wall above, there is deliberately no room to arc a forward turn within the
        // pocket - narrower than any physical U-turn.
        if (episodeBootstrapStage == BootstrapStage.DeadEnd && carController.gridManager != null)
        {
            var gridManager = carController.gridManager;
            float cellSize = gridManager.cellSize;

            for (int d = 1; d <= deadEndDepthCells; d++)
            {
                Vector3 forwardOffset = transform.forward * (d * cellSize);
                gridManager.SetTileTypeAt(spawnPos + forwardOffset - transform.right * cellSize, TileType.Terminal);
                gridManager.SetTileTypeAt(spawnPos + forwardOffset + transform.right * cellSize, TileType.Terminal);
            }
            Vector3 capCenter = spawnPos + transform.forward * ((deadEndDepthCells + 1) * cellSize);
            gridManager.SetTileTypeAt(capCenter, TileType.Terminal);
            gridManager.SetTileTypeAt(capCenter - transform.right * cellSize, TileType.Terminal);
            gridManager.SetTileTypeAt(capCenter + transform.right * cellSize, TileType.Terminal);
        }

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
    // reverse gear - see the flipped approach-bonus dot in OnTriggerEnter). DeadEnd: goal behind,
    // beyond a pocket that physically requires reversing straight out first (see OnEpisodeBegin;
    // no approach-bonus flip needed here - the final approach into the goal is a normal forward
    // drive once the car has backed clear of the pocket). Side: randomised angle/distance for
    // variety, bridging toward the eventual full-random placement.
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

            case BootstrapStage.DeadEnd:
            {
                float minDist = ep.GetWithDefault("bootstrap_deadend_goal_distance_min", 10f);
                float maxDist = ep.GetWithDefault("bootstrap_deadend_goal_distance_max", 18f);
                float angleSpread = ep.GetWithDefault("bootstrap_deadend_goal_angle_spread_deg", 60f);
                do
                {
                    // 180 = straight behind; +/- angleSpread keeps it safely in the rear
                    // hemisphere, away from the pocket built straight ahead of the spawn.
                    float angle = 180f + Random.Range(-angleSpread, angleSpread);
                    float distance = Random.Range(minDist, Mathf.Max(minDist, maxDist));
                    Vector3 dir = Quaternion.AngleAxis(angle, Vector3.up) * transform.forward;
                    goalPos = spawnPos + dir * distance;
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
        sensor.AddObservation(carController.accelerationInput + carController.brakeInput); // 1 - pedal magnitude (only one is ever nonzero)

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
        // so the agent can no longer press acceleration and brake at once (car_component previously allowed it).
        carController.accelerationInput = pedalIsBrake ? 0f : pedal;
        carController.brakeInput = pedalIsBrake ? pedal : 0f;
        carController.RequestReverseGear(actions.DiscreteActions[0] == 1);

        episodeStepCount++;
        float scaledTimePenalty = rwTimePenalty * (1f + episodeStepCount * rwTimePenaltyGrowth);
        AddReward(scaledTimePenalty);

        // Escalating idle penalty: speed (not distance-to-goal) is the signal, specifically so
        // this can never penalise backing away from the goal the way a distance-delta shaping
        // reward would - only genuine standing-still counts. idleGraceSteps lets a real stop
        // (e.g. mid gear-shift) pass without cost; past that, the penalty grows the longer it
        // persists instead of staying flat, so a frozen policy can't just tolerate it indefinitely.
        if (idlePenaltyEnabled)
        {
            float speed = rb.linearVelocity.magnitude;
            idleStepCounter = speed < idleSpeedThreshold ? idleStepCounter + 1 : 0;

            int stepsPastGrace = idleStepCounter - idleGraceSteps;
            if (stepsPastGrace > 0)
                AddReward(rwIdlePenaltyBase * (1f + stepsPastGrace * rwIdlePenaltyGrowth));
        }

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
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActions = actionsOut.ContinuousActions;
        var discreteActions = actionsOut.DiscreteActions;
        Keyboard kb = Keyboard.current;
        Gamepad gp = Gamepad.current;

        float horizontal = 0f;
        float acceleration = 0f;
        float brake = 0f;

        if (kb != null)
        {
            if (kb.aKey.isPressed) horizontal -= 1f;
            if (kb.dKey.isPressed) horizontal += 1f;
            if (kb.wKey.isPressed) acceleration = 1f;
            if (kb.sKey.isPressed) brake = 1f;
        }

        // Controller: left stick + analog triggers give true continuous values,
        // unlike the keyboard's hard 0/1 steps — combined with keyboard via Max/add so either device works.
        if (gp != null)
        {
            horizontal += gp.leftStick.x.ReadValue();
            acceleration = Mathf.Max(acceleration, gp.rightTrigger.ReadValue());
            brake = Mathf.Max(brake, gp.leftTrigger.ReadValue());
        }

        // The action space only allows one pedal at a time; brake wins if both are held
        // (e.g. W+S together), since braking is the safer default to fall back to.
        bool brakePedal = brake > 0f;
        float pedal = brakePedal ? brake : acceleration;

        continuousActions[0] = Mathf.Clamp(horizontal, -1f, 1f);
        // OnActionReceived expects ContinuousActions[1] in the same -1..1 range a trained policy's
        // tanh-squashed continuous output uses, remapping -1->0 and +1->1 pedal. pedal here is
        // already 0..1 (no press -> full press), so it must be rescaled to match, not passed
        // through Clamp01 directly - otherwise idle (pedal=0) becomes ContinuousActions[1]=0, which
        // OnActionReceived remaps to a phantom pedal=0.5 instead of the intended 0.
        continuousActions[1] = Mathf.Clamp(pedal * 2f - 1f, -1f, 1f);

        // Echo the current gear (driven by the Q-toggle in Update()) rather than requesting
        // a fresh value here — otherwise RequestReverseGear() in OnActionReceived would fight
        // the toggle every physics step.
        discreteActions[0] = carController.reverseGear ? 1 : 0;
        discreteActions[1] = brakePedal ? 1 : 0;
    }

    void OnTriggerEnter(Collider other)
    {
        if (other.transform == goal && !goalTriggeredThisEpisode)
        {
            goalTriggeredThisEpisode = true;
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
