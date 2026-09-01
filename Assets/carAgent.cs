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

    // Fixed-map trajectory-comparison eval mode (see eval_trajectories.py) - when enabled via the
    // fixed_eval_enabled environment parameter, OnEpisodeBegin skips all the normal random spawn/
    // goal placement below and uses fixed_spawn_x/z, fixed_goal_x/z, fixed_spawn_heading_deg
    // instead, and CollectObservations reports Custom/PosX/Custom/PosZ/Custom/ReverseGear every
    // decision step so the eval script can reconstruct the exact path driven (and tell a genuine
    // reverse maneuver apart from a tight forward turn). Fields (not locals) since both
    // OnEpisodeBegin and CollectObservations need to read them. Never set during ordinary
    // training/eval (mlagents-learn and eval_checkpoints.py never touch this parameter), so this
    // mode is fully opt-in and costs nothing when off.
    private bool fixedEvalEnabled = false;
    private float fixedSpawnHeadingDeg = 0f;

    [Tooltip("Adds one extra observation - fraction of this episode's step budget remaining " +
        "(1 = just started, 0 = about to hit MaxStep) - so the policy can actually condition on " +
        "how close it is to a MaxStep timeout (e.g. useful alongside reward_maxstep_penalty, which " +
        "otherwise has no way to be anticipated). CHANGES THE OBSERVATION VECTOR SIZE, which is " +
        "baked into the trained network's input layer - this cannot be toggled on/off for an " +
        "already-trained model/build, only decided BEFORE training (set here, then bump this " +
        "Agent's BehaviorParameters -> Vector Observation Space Size by +1 to match, then build). " +
        "To compare with/without, build and train two separate binaries, one with each setting.")]
    public bool includeRemainingStepsObservation = false;

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

    // Tracks the tile the car was on as of the previous decision step, so CollectObservations can
    // report a "visit" (Custom/TileVisit<Type>) only on a genuine change of surface, not once per
    // step the way Custom/TileTime<Type> already does - answers "how often does the policy cross
    // onto each surface" independent of how long it then lingers there (a slow surface like
    // Gravel/Slippery racks up more TileTime just by physically taking longer to cross, even at
    // the same crossing frequency as Asphalt). Null at the start of each episode (reset in
    // OnEpisodeBegin) specifically so the very first CollectObservations call of the episode - the
    // car's starting tile - always counts as a visit too, not just later changes.
    private TileType? previousTileType = null;

    // Per-episode accumulators for the same two tile stats, reported once at the START of the
    // NEXT OnEpisodeBegin (for the episode that just ended, before being reset below) as
    // Custom/EpisodeTileTime<Type>/Custom/EpisodeTileVisit<Type> - one Sum-aggregated value per
    // type PER EPISODE, unlike the step-level Custom/TileTime</Visit<Type> stats above (which stay
    // exactly as they were - this is purely additive). Gives eval_checkpoints.py one list entry
    // per episode per type, so it can take a genuine std-across-episodes the same way it already
    // does for episode_rewards/episode_lengths. Indexed by (int)TileType (5 contiguous values, see
    // TileType.cs) rather than a Dictionary, to avoid a new using directive for one small
    // fixed-size lookup.
    private readonly float[] episodeTileTimeSteps = new float[5];
    private readonly int[] episodeTileVisits = new int[5];
    // False only before the run's first episode has actually happened, so that first OnEpisodeBegin
    // doesn't report a bogus all-zero "episode" - same guard shape as hasBootstrapStageHistory above.
    private bool hasEpisodeTileStatsToReport = false;

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
    private float rwMaxStepPenalty;
    private float rwTimePenaltyGrowth;
    private float rwSensorTerminalPenalty;
    private float sensorSeverityExponent = 1f;

    // Dense forward-sensor terminal-avoidance shaping state (see UpdateRewardWeights/
    // CollectObservations, car_agent.yaml's sensor_terminal_lagrangian_* block) - severity is
    // always in [0, 1] (CarSensor.GetNearestForwardLineTerminalSeverity), so -1f is a safe "no
    // reading yet this episode" sentinel, reset in OnEpisodeBegin. The very first decision step of
    // an episode only seeds this baseline, no reward fires (there's no genuine "previous" state to
    // compare against yet).
    private float previousForwardLineTerminalSeverity = -1f;

    // Rank (0=nearest, see CarSensor.GetNearestForwardLineTerminalRank) of the nearest dead-ahead-
    // line sensor reading Terminal as of the last CollectObservations call, or -1 if none were red.
    // Tracked alongside previousForwardLineTerminalSeverity (updated every step regardless of
    // whether that step's reward fires) purely so OnEpisodeBegin can gate the MaxStep-outcome
    // handling on RANK rather than severity value - see episodeCumulativeSensorSeverity's comment
    // and CarSensor.GetNearestForwardLineTerminalRank's own comment for why rank stays correct
    // across exponent changes when a fixed severity threshold wouldn't.
    private int previousForwardLineTerminalRank = -1;

    // Running per-episode sum of the SAME (phi_now - phi_previous) deltas the reward shaping above
    // uses, but UNSCALED by rwSensorTerminalPenalty (lambda) - deliberately lambda-independent, so
    // the Lagrangian dual-ascent's target-tracking metric doesn't itself inflate as lambda rises
    // (see the Custom/ForwardLineTerminalExposure reporting in OnEpisodeBegin, which reads this).
    // Telescopes to (phi_at_episode_end - phi_at_first_reading) by construction, same as the reward
    // shaping's own telescoping property - a car that approaches danger and fully recovers nets to
    // ~0 here (not some nonzero cumulative "time spent near danger"), matching the design intent
    // that only the NET/final outcome should count, not transient exposure that gets corrected.
    // Reset to 0 in OnEpisodeBegin, alongside previousForwardLineTerminalSeverity.
    private float episodeCumulativeSensorSeverity = 0f;

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

        // Applied once, the step a MaxStepReached timeout fires (see FixedUpdate) - on top of,
        // not instead of, the flat per-step reward_time_penalty/idle penalty already accumulated
        // over the episode. 0 by default (no behavior change); purely a manual override now - the
        // adaptive (Lagrangian dual-ascent) auto-tuning that used to drive this was removed, it
        // never converged to a useful policy.
        rwMaxStepPenalty = ep.GetWithDefault("reward_maxstep_penalty", 0.0f);

        // Dense forward-sensor terminal-avoidance shaping (potential-based) - pushed by
        // trainer_controller_patched.py's "sensor_terminal" Lagrangian constraint, same
        // already-negative "penalty value" convention as reward_terminal_penalty/rwTerminalPenalty
        // above (0 = inert, no behavior change, unless that mechanism is bound and enabled). See
        // car_agent.yaml's sensor_terminal_lagrangian_* block and CollectObservations() below for
        // the full mechanism.
        rwSensorTerminalPenalty = ep.GetWithDefault("sensor_terminal_penalty_lambda", 0.0f);

        // See CarSensor.GetNearestForwardLineTerminalSeverity's exponent parameter comment - 1.0
        // (default) is the original plain linear fraction, unchanged from before this parameter
        // existed. Read here (not in CarSensor) since CarSensor has no ML-Agents dependency by design.
        sensorSeverityExponent = ep.GetWithDefault("sensor_severity_exponent", 1.0f);

        // Episode time budget (built-in Agent.MaxStep, read fresh each episode). Defaults to
        // 5000, matching the Inspector's static value, so leaving max_step_budget out of the yaml
        // changes nothing. Works as a plain fixed override on its own; trainer_controller_patched
        // .py's adaptive_step_budget_enabled mechanism can additionally drive it - see
        // car_agent.yaml's "Adaptive step-time budget" block, which is NOT the same technique as
        // the Lagrangian-style penalties above despite living in the same neighbourhood (it
        // adjusts the episode's actual task structure, not a reward coefficient).
        MaxStep = Mathf.RoundToInt(ep.GetWithDefault("max_step_budget", 5000f));

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
            AddReward(rwMaxStepPenalty);
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
        // Guarded by Application.isEditor so a stray true value left on the Inspector can never
        // silently reach a real build again - these two are documented as editor-only testing
        // conveniences (see their tooltips), but neither was actually gated on being in the
        // Editor, so a scene saved with forceStageInEditor=true baked it into every build,
        // permanently forcing every episode (training AND eval) into a single fixed bootstrap
        // stage regardless of bootstrap_curriculum_enabled - this was live in the checked-in
        // scene (forceStageInEditor=1, editorForcedStage=3/DeadEnd) and explains the CSV symptoms
        // that led here: 100% Custom/TileTimeGravel=0 (forceAllNormal blanking the whole grid),
        // ~14-16 tick average episodes (crashing out of the walled DeadEnd pocket), and 0% goal
        // rate even at ~13M/30M training steps.
        if (Application.isEditor)
        {
            if (forceStageInEditor) return editorForcedStage;
            if (disableCurriculumInEditor) return BootstrapStage.None;
        }

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

        // Report the just-finished episode's cumulative sensor severity BEFORE resetting the
        // accumulator below - see episodeCumulativeSensorSeverity's field comment for the
        // lambda-independence rationale. Three-way outcome logic:
        //   Terminated - forced to 1.0 (worst) REGARDLESS of what episodeCumulativeSensorSeverity
        //     actually accumulated - the nearest forward-line sensor point sits 3m ahead of the car
        //     (not at its own position), so a crash via wall-tag collision, flipping, or reversing
        //     into a hazard wouldn't reliably show a red forward-line reading even though it's
        //     exactly the outcome this metric should flag as worst-case.
        //   MaxStep, nearest red sensor at rank 0 or 1 (nearest or 2nd-nearest of the 7) - uses the
        //     actual last-recorded severity directly (NOT the telescoped cumulative) - a policy that
        //     times out while frozen right in front of a hazard is a real failure the telescoped net
        //     can hide (e.g. it oscillated near the hazard all episode and just happened to be at a
        //     lower reading the exact instant the clock ran out - net looks like "handled it fine"
        //     even though it never actually resolved anything). Gated on RANK, not a fixed severity
        //     threshold, so this means the same thing regardless of sensor_severity_exponent.
        //   MaxStep, nearest red at rank >= 2 or no red sensor at all - 0 ("probably wasn't the
        //     hazard's fault" that it timed out - too far away to plausibly be why).
        //   Goal - the telescoped net cumulative (episodeCumulativeSensorSeverity), unchanged -
        //     successfully reaching the goal after passing near a hazard earlier in the episode is a
        //     genuinely handled outcome, so this stays "goes to danger and back nets to ~0".
        // hasBootstrapStageHistory guards "has a previous episode actually happened" (same role as
        // hasEpisodeTileStatsToReport below, reused here since it's already set at this exact point
        // in the very first episode). Keeps the OLD stat name (Custom/ForwardLineTerminalExposure) -
        // previously reported every decision step as the instantaneous reading, now reported once
        // per episode as this three-way value - so stats_patched.py/trainer_controller_patched.py's
        // dual-ascent wiring and sensor_terminal_target_exposure keep working with zero Python-side
        // changes, same "redefine the meaning, keep the name" pattern as
        // GetNearestForwardLineTerminalSeverity() itself.
        if (hasBootstrapStageHistory && carSensor != null && carSensor.ForwardLineSensorCount > 0)
        {
            float episodeEndSeverity;
            if (lastEpisodeEndReason == "Terminated")
            {
                episodeEndSeverity = 1f;
            }
            else if (lastEpisodeEndReason == "MaxStep" && previousForwardLineTerminalRank >= 0 && previousForwardLineTerminalRank <= 1)
            {
                episodeEndSeverity = Mathf.Clamp01(previousForwardLineTerminalSeverity);
            }
            else if (lastEpisodeEndReason == "MaxStep")
            {
                episodeEndSeverity = 0f;
            }
            else
            {
                episodeEndSeverity = Mathf.Clamp01(episodeCumulativeSensorSeverity);
            }
            Academy.Instance.StatsRecorder.Add("Custom/ForwardLineTerminalExposure", episodeEndSeverity, StatAggregationMethod.Average);
        }
        previousForwardLineTerminalSeverity = -1f;
        previousForwardLineTerminalRank = -1;
        episodeCumulativeSensorSeverity = 0f;

        // Report the just-finished episode's per-tile-type totals before resetting them for the
        // new episode - see the field comments above for why this lives here rather than at each
        // of the several episode-ending call sites (Goal/Terminated/MaxStep) scattered below.
        if (hasEpisodeTileStatsToReport)
        {
            for (int i = 0; i < episodeTileTimeSteps.Length; i++)
            {
                var t = (TileType)i;
                Academy.Instance.StatsRecorder.Add(
                    $"Custom/EpisodeTileTime{t}", episodeTileTimeSteps[i], StatAggregationMethod.Sum);
                Academy.Instance.StatsRecorder.Add(
                    $"Custom/EpisodeTileVisit{t}", episodeTileVisits[i], StatAggregationMethod.Sum);
            }
        }
        for (int i = 0; i < episodeTileTimeSteps.Length; i++)
        {
            episodeTileTimeSteps[i] = 0f;
            episodeTileVisits[i] = 0;
        }
        hasEpisodeTileStatsToReport = true;

        rb.linearVelocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;
        episodeStepCount = 0;
        idleStepCounter = 0;
        previousTileType = null;

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
            // Off (train pool) unless a connected client explicitly asks for the held-out set -
            // eval_checkpoints.py sets this to force validation-only maps, so evaluation results
            // reflect generalisation rather than memorisation of the training pool. mlagents-learn
            // never sets it, so ordinary training always sees Maps/VoronoiTrain.
            carController.gridManager.useValidationMaps =
                Academy.Instance.EnvironmentParameters.GetWithDefault("use_validation_maps", 0f) >= 0.5f;

            // See the fixedEvalEnabled field comment above - eval_trajectories.py sets these two
            // to pin a specific pre-baked map instead of GridManager's usual random pick.
            // Ordinary training/eval never sets fixed_eval_enabled, so forcedMapIndex stays at
            // its default -1 (unchanged random behavior) for every other code path.
            fixedEvalEnabled =
                Academy.Instance.EnvironmentParameters.GetWithDefault("fixed_eval_enabled", 0f) >= 0.5f;
            carController.gridManager.forcedMapIndex = fixedEvalEnabled
                ? Mathf.RoundToInt(Academy.Instance.EnvironmentParameters.GetWithDefault("fixed_map_index", -1f))
                : -1;

            carController.gridManager.Regenerate();
        }
        else
        {
            fixedEvalEnabled =
                Academy.Instance.EnvironmentParameters.GetWithDefault("fixed_eval_enabled", 0f) >= 0.5f;
        }

        Vector3 spawnPos;
        if (fixedEvalEnabled)
        {
            float fixedSpawnX = Academy.Instance.EnvironmentParameters.GetWithDefault("fixed_spawn_x", 0f);
            float fixedSpawnZ = Academy.Instance.EnvironmentParameters.GetWithDefault("fixed_spawn_z", 0f);
            spawnPos = new Vector3(fixedSpawnX, startPosition.y, fixedSpawnZ);
            fixedSpawnHeadingDeg = Academy.Instance.EnvironmentParameters.GetWithDefault("fixed_spawn_heading_deg", 0f);
        }
        else
        {
            int attempts = 0;
            do {
                float spawnX = Random.Range(spawnAreaMin.x, spawnAreaMax.x);
                float spawnZ = Random.Range(spawnAreaMin.y, spawnAreaMax.y);
                spawnPos = new Vector3(spawnX, startPosition.y, spawnZ);
            } while (!IsSpawnableTile(spawnPos) && ++attempts < 100);
        }
        transform.position = spawnPos;

        // On Voronoi maps, once bootstrap has finished (forceAllNormal - see above - would
        // otherwise blanket the whole grid in Asphalt and make every direction look equally
        // "road"), face the car along the local road direction instead of a fully random yaw, so
        // it starts out driving down the road like a real car rather than parked sideways/facing
        // off into the gravel shoulder. In fixed-eval mode, ComputeRoadAlignedHeading's internal
        // coin-flip between the two facings along a road (see its own comment) would otherwise
        // make the deterministic and stochastic rollouts of the SAME candidate start facing
        // opposite directions on different runs - a spurious source of "divergence" that's really
        // just a different starting orientation, not a policy difference - so fixed-eval always
        // uses the externally-supplied fixedSpawnHeadingDeg instead.
        bool useRoadAlignedHeading = !fixedEvalEnabled
            && carController.gridManager != null
            && carController.gridManager.mapSource == MapSource.Voronoi
            && episodeBootstrapStage == BootstrapStage.None;
        transform.rotation = Quaternion.Euler(
            0f,
            fixedEvalEnabled ? fixedSpawnHeadingDeg
                : (useRoadAlignedHeading ? ComputeRoadAlignedHeading(spawnPos) : Random.Range(0f, 360f)),
            0f);

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
        // forward. Placed AFTER the goal so it can't affect the goal's own IsSpawnableTile retry loop.
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

    bool IsSpawnableTile(Vector3 worldPos)
    {
        if (carController.gridManager == null) return true;
        return carController.gridManager.GetTileAt(worldPos) == carController.gridManager.SpawnableTileType;
    }

    // Samples the 5x5 neighbourhood of Asphalt cells around worldPos (GridManager.
    // GetNeighborhood5x5 - same helper CarSensor readings ultimately bottom out on) and averages
    // their offset angles with the standard "doubled angle" circular-mean trick: a road segment is
    // a line, not an arrow, so it's 180-degree ambiguous (facing either way along it is equally
    // "aligned") - a plain vector average of offsets pointing opposite directions along a straight
    // road would cancel to ~zero, but doubling each angle before averaging (then halving the
    // result) correctly finds the dominant *line* orientation instead. Falls back to a uniformly
    // random heading when too few Asphalt neighbours are found to give a reliable direction (e.g.
    // an isolated patch), then picks one of the two directions along the road at random.
    float ComputeRoadAlignedHeading(Vector3 worldPos)
    {
        var neighborhood = new TileType[25];
        carController.gridManager.GetNeighborhood5x5(worldPos, neighborhood);

        float sumSin = 0f, sumCos = 0f;
        int count = 0;
        int i = 0;
        for (int dz = -2; dz <= 2; dz++)
        {
            for (int dx = -2; dx <= 2; dx++)
            {
                if ((dx != 0 || dz != 0) && neighborhood[i] == TileType.Asphalt)
                {
                    float angleRad = Mathf.Atan2(dx, dz); // matches Unity's yaw convention (0 deg = +Z)
                    sumSin += Mathf.Sin(2f * angleRad);
                    sumCos += Mathf.Cos(2f * angleRad);
                    count++;
                }
                i++;
            }
        }

        if (count < 2)
            return Random.Range(0f, 360f);

        float roadHeadingDeg = Mathf.Atan2(sumSin, sumCos) * 0.5f * Mathf.Rad2Deg;
        return roadHeadingDeg + (Random.value < 0.5f ? 0f : 180f); // face either way along the road
    }

    // Bootstrap curriculum takes placement priority while active; once it's finished (or
    // disabled), goal placement is pure full-random across the whole goal area - except under
    // mapSource=Voronoi, where it's additionally capped to voronoiNearGoalMaxDistance (see
    // VoronoiNearGoalMaxDistance below) for the first voronoi_near_goal_steps GLOBAL steps. Unlike
    // Perlin's open blobs, a genuinely random point on a curving road network can be very far from
    // spawn the instant bootstrap ends - a sudden difficulty spike right after the controls-only
    // curriculum - so this ramps the cap from tight to unconstrained instead of dropping the agent
    // straight into the hardest case.
    Vector3 PickGoalPosition(Vector3 spawnPos)
    {
        if (fixedEvalEnabled)
        {
            float fixedGoalX = Academy.Instance.EnvironmentParameters.GetWithDefault("fixed_goal_x", 0f);
            float fixedGoalZ = Academy.Instance.EnvironmentParameters.GetWithDefault("fixed_goal_z", 0f);
            return new Vector3(fixedGoalX, goal.position.y, fixedGoalZ);
        }
        if (episodeBootstrapStage != BootstrapStage.None)
            return PickBootstrapGoalPosition(spawnPos);

        float maxDistance = VoronoiNearGoalMaxDistance();

        Vector3 goalPos;
        int attempts = 0;
        do
        {
            if (float.IsPositiveInfinity(maxDistance))
            {
                float goalX = Random.Range(goalAreaMin.x, goalAreaMax.x);
                float goalZ = Random.Range(goalAreaMin.y, goalAreaMax.y);
                goalPos = new Vector3(goalX, goal.position.y, goalZ);
            }
            else
            {
                // Sample within a maxDistance disk around spawn directly, rather than uniformly
                // across the whole goal area and rejecting anything too far - with a tight early
                // cap plus the Asphalt-only requirement, rejection sampling over the full area
                // would rarely land a valid point within 100 attempts.
                Vector2 offset = Random.insideUnitCircle * maxDistance;
                goalPos = new Vector3(
                    Mathf.Clamp(spawnPos.x + offset.x, goalAreaMin.x, goalAreaMax.x),
                    goal.position.y,
                    Mathf.Clamp(spawnPos.z + offset.y, goalAreaMin.y, goalAreaMax.y));
            }
        } while (!IsSpawnableTile(goalPos) && ++attempts < 100);
        return goalPos;
    }

    // Total length of the bootstrap curriculum (Front+Behind+DeadEnd+Side), in the same "GLOBAL
    // steps / training_num_envs, corrected for DecisionPeriod" units as Academy.TotalStepCount -
    // same convention/formula as DetermineBootstrapStage(), computed independently here so
    // VoronoiNearGoalMaxDistance can anchor its own ramp to "steps since bootstrap ended" without
    // depending on DetermineBootstrapStage having already run earlier in the same OnEpisodeBegin.
    // Returns 0 when bootstrap_curriculum_enabled is off, so the voronoi ramp then starts counting
    // from step 0 - correct, since there's no bootstrap period to wait out in that case.
    float BootstrapTotalSteps()
    {
        var ep = Academy.Instance.EnvironmentParameters;
        if (ep.GetWithDefault("bootstrap_curriculum_enabled", 1f) < 0.5f) return 0f;

        float numEnvs = Mathf.Max(1f, ep.GetWithDefault("training_num_envs", 1f));
        float decisionPeriod = decisionRequester != null ? Mathf.Max(1, decisionRequester.DecisionPeriod) : 1f;
        float frontSteps   = ep.GetWithDefault("bootstrap_stage_front_steps", 20_000f) * decisionPeriod / numEnvs;
        float behindSteps  = ep.GetWithDefault("bootstrap_stage_behind_steps", 20_000f) * decisionPeriod / numEnvs;
        float deadEndSteps = ep.GetWithDefault("bootstrap_stage_deadend_steps", 20_000f) * decisionPeriod / numEnvs;
        float sideSteps    = ep.GetWithDefault("bootstrap_stage_side_steps", 20_000f) * decisionPeriod / numEnvs;
        return frontSteps + behindSteps + deadEndSteps + sideSteps;
    }

    // Returns +infinity (no cap) unless mapSource=Voronoi and voronoi_near_goal_enabled is set,
    // in which case it linearly ramps from voronoi_near_goal_distance_start to _end over
    // voronoi_near_goal_steps GLOBAL steps - same "GLOBAL steps / training_num_envs" convention as
    // every other step-based curriculum in this file (see UpdateRewardWeights/DetermineBootstrapStage).
    // Anchored to steps SINCE BOOTSTRAP ENDED (Academy.TotalStepCount - BootstrapTotalSteps()), not
    // raw TotalStepCount - the ramp only ever matters once bootstrap stops overriding goal placement
    // (see PickGoalPosition), so counting from absolute step 0 was silently burning part of the ramp
    // during bootstrap itself: e.g. bootstrap=1.5M/voronoi=3M meant the cap was already at progress
    // 0.5 (not the intended tight 15-unit start) the instant bootstrap ended, and the ramp finished
    // 1.5M steps early - exactly the "sudden difficulty spike right after bootstrap" this mechanism
    // was built to avoid. Anchoring here restores the intended sequential handoff: full ramp length,
    // starting fresh right when full-random placement actually begins.
    float VoronoiNearGoalMaxDistance()
    {
        if (carController.gridManager == null || carController.gridManager.mapSource != MapSource.Voronoi)
            return float.PositiveInfinity;

        var ep = Academy.Instance.EnvironmentParameters;
        if (ep.GetWithDefault("voronoi_near_goal_enabled", 0f) < 0.5f)
            return float.PositiveInfinity;

        float numEnvs = Mathf.Max(1f, ep.GetWithDefault("training_num_envs", 1f));
        float decisionPeriod = decisionRequester != null ? Mathf.Max(1, decisionRequester.DecisionPeriod) : 1f;
        float rampSteps = ep.GetWithDefault("voronoi_near_goal_steps", 5_000_000f) * decisionPeriod / numEnvs;
        float stepsSinceBootstrapEnd = Mathf.Max(0f, Academy.Instance.TotalStepCount - BootstrapTotalSteps());
        float progress = rampSteps > 0f ? Mathf.Clamp01(stepsSinceBootstrapEnd / rampSteps) : 1f;

        float distStart = ep.GetWithDefault("voronoi_near_goal_distance_start", 15f);
        float distEnd = ep.GetWithDefault("voronoi_near_goal_distance_end", 100f);
        return Mathf.Lerp(distStart, distEnd, progress);
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
                } while (!IsSpawnableTile(goalPos) && ++attempts < 100);
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
                } while (!IsSpawnableTile(goalPos) && ++attempts < 100);
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
                } while (!IsSpawnableTile(goalPos) && ++attempts < 100);
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
                } while (!IsSpawnableTile(goalPos) && ++attempts < 100);
                return goalPos;
            }
        }
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        carSensor?.UpdateReadings();

        // Dense forward-sensor terminal-avoidance shaping - see car_agent.yaml's
        // sensor_terminal_lagrangian_* block for the mechanism/derivation this still follows.
        // Phi(s) is now a PROXIMITY-based severity, not the old raw fraction-of-7-sensors-red: it's
        // driven by whichever dead-ahead-line sensor currently reading Terminal is NEAREST the car
        // (CarSensor.GetNearestForwardLineTerminalSeverity - severity 1.0 if the closest point
        // itself is red, down to 1/7 if only the farthest point is, 0 if none are) - a hazard read
        // by the 3m point is far more urgent than one only visible at 21m, which the old plain
        // count couldn't distinguish (2 hazards read scored the same regardless of whether they
        // were the two nearest or two farthest points). Reward this step = rwSensorTerminalPenalty
        // * (Phi(s_now) - Phi(s_previous)) - UNCHANGED potential-based-shaping structure/telescoping
        // property from before, only what Phi measures changed. rwSensorTerminalPenalty is already
        // negative (same convention as rwTerminalPenalty) when the mechanism is active, so Phi
        // INCREASING (a nearer sensor just went red, i.e. approaching/already-closer danger) is
        // penalized and Phi DECREASING (retreating to only-farther-or-no hazard) is rewarded, by the
        // same magnitude. Fan-shape-only (ForwardLineSensorCount is 0 for Stadium/Circle, where this
        // silently no-ops).
        //
        // Custom/ForwardLineTerminalExposure is no longer reported here per-step - see OnEpisodeBegin,
        // where it's now reported ONCE per just-ended episode as the net accumulated severity
        // (episodeCumulativeSensorSeverity), not the instantaneous per-step reading. Any
        // sensor_terminal_target_exposure value calibrated against either the old count-based metric
        // OR the earlier per-step-average proximity metric is not a sane target under this episode-
        // cumulative one - recalibrate against a fresh baseline run before trusting an old target
        // number here again.
        if (carSensor != null && carSensor.ForwardLineSensorCount > 0)
        {
            float phi = carSensor.GetNearestForwardLineTerminalSeverity(sensorSeverityExponent);
            if (previousForwardLineTerminalSeverity >= 0f)
            {
                float delta = phi - previousForwardLineTerminalSeverity;
                AddReward(rwSensorTerminalPenalty * delta);
                episodeCumulativeSensorSeverity += delta;
            }
            previousForwardLineTerminalSeverity = phi;
            previousForwardLineTerminalRank = carSensor.GetNearestForwardLineTerminalRank();
        }

        // One Sum-aggregated stat per decision step, tagged by whichever tile the car is
        // currently on - same StatsRecorder pattern as Custom/GoalReached etc. above, so this
        // shows up both live in TensorBoard during training and per-checkpoint in
        // eval_checkpoints.py's StatsSideChannel readback (see run_checkpoint there). Answers
        // "does the policy actually favour Asphalt, or drive straight through Gravel/blob
        // regions regardless of what's ahead" - divide each Custom/TileTime<Type> by their sum
        // for the fraction of time spent on that surface.
        Academy.Instance.StatsRecorder.Add(
            $"Custom/TileTime{carController.currentTileType}", 1f, StatAggregationMethod.Sum);
        episodeTileTimeSteps[(int)carController.currentTileType] += 1f;

        // One Sum-aggregated stat per genuine surface change (including the episode's starting
        // tile - previousTileType is null only right after OnEpisodeBegin) - unlike TileTime
        // above, this doesn't grow just because the car sat on a slow surface longer, so it's a
        // speed-independent count of how often each surface is actually crossed onto.
        if (previousTileType == null || previousTileType.Value != carController.currentTileType)
        {
            Academy.Instance.StatsRecorder.Add(
                $"Custom/TileVisit{carController.currentTileType}", 1f, StatAggregationMethod.Sum);
            episodeTileVisits[(int)carController.currentTileType] += 1;
            previousTileType = carController.currentTileType;
        }

        // Raw per-decision-step (x, z) position stream, for eval_trajectories.py's path-comparison
        // plots - ONLY reported in fixed-eval mode (see fixedEvalEnabled/OnEpisodeBegin above).
        // Unconditional reporting (like TileTime/TileVisit) would be meaningless during ordinary
        // parallel training/eval, where many maps/episodes' positions would interleave in the same
        // StatsSideChannel list with no way to tell them apart - fixed-eval mode is deliberately
        // restricted to one agent/one episode at a time so this stream stays unambiguous.
        // MostRecent (not Sum) since this is a raw coordinate, not something to accumulate.
        // Custom/ReverseGear rides along the same stream (same cadence, same MostRecent semantics)
        // so eval_trajectories.py can tell a genuine reverse maneuver apart from a tight forward
        // turn/near-U-turn - both look like a loop in (x, z) alone, position data can't distinguish
        // them without this.
        if (fixedEvalEnabled)
        {
            Academy.Instance.StatsRecorder.Add("Custom/PosX", transform.position.x, StatAggregationMethod.MostRecent);
            Academy.Instance.StatsRecorder.Add("Custom/PosZ", transform.position.z, StatAggregationMethod.MostRecent);
            Academy.Instance.StatsRecorder.Add("Custom/ReverseGear", carController.reverseGear ? 1f : 0f, StatAggregationMethod.MostRecent);
        }

        Vector3 toGoal = goal.position - transform.position;
        sensor.AddObservation(transform.InverseTransformDirection(toGoal.normalized)); // 3
        sensor.AddObservation(toGoal.magnitude);                                        // 1
        sensor.AddObservation(transform.InverseTransformDirection(rb.linearVelocity)); // 3
        sensor.AddObservation(transform.InverseTransformDirection(rb.angularVelocity).y); // 1 - yaw rate
        sensor.AddObservation(carController.currentSteerAngle / carController.maxSteerAngle); // 1
        sensor.AddObservation(carController.reverseGear ? 1f : 0f); // 1
        sensor.AddObservation(pedalIsBrake ? 1f : 0f); // 1 - which pedal is selected
        sensor.AddObservation(carController.accelerationInput + carController.brakeInput); // 1 - pedal magnitude (only one is ever nonzero)

        // Optional: fraction of this episode's MaxStep budget remaining, 1=start -> 0=timeout.
        // Gated behind includeRemainingStepsObservation - see its field comment above for why this
        // must be decided before training, not toggled at runtime. When off (default), this adds
        // nothing and the observation vector is unchanged from before.
        if (includeRemainingStepsObservation)
        {
            float remainingStepsFraction = MaxStep > 0 ? Mathf.Clamp01(1f - (float)StepCount / MaxStep) : 1f;
            sensor.AddObservation(remainingStepsFraction); // 1 (only present if includeRemainingStepsObservation)
        }

        // Current tile type, one-hot (5 values - see TileType.cs)                       // 5
        AddTileOneHot(sensor, carController.currentTileType);

        // sensor readings: carSensor.SensorCount × 5 values (one-hot per point)
        // If the sensor shape/radius/ring settings change, update Space Size to: 17 + SensorCount * 5
        // (+1 more if includeRemainingStepsObservation is enabled)
        if (carSensor != null && carSensor.readings != null)
        {
            foreach (var t in carSensor.readings)
                AddTileOneHot(sensor, t);
        }
        else
        {
            // fallback: 13 zeros per reading × 5 (one-hot) for default circular radius=2
            int count = carSensor != null ? carSensor.SensorCount : 13;
            for (int i = 0; i < count * 5; i++)
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
        sensor.AddObservation(type == TileType.Slippery     ? 1f : 0f);
        sensor.AddObservation(type == TileType.SpeedLimited ? 1f : 0f);
        sensor.AddObservation(type == TileType.Terminal     ? 1f : 0f);
        sensor.AddObservation(type == TileType.Asphalt      ? 1f : 0f);
        sensor.AddObservation(type == TileType.Gravel       ? 1f : 0f);
    }
}
