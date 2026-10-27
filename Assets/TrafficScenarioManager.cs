using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using Unity.MLAgents;

// Owns a shared map and a shared clock. CarAgent retains its original single-car
// behavior when no TrafficScenarioManager is assigned.
public class TrafficScenarioManager : MonoBehaviour
{
    public const string MeasurementSchema = "multi-car-random-placement-v4";
    public const string FixedMeasurementSchema = "finish-tile-and-per-seat-tile-seconds-v2";
    private const int TileTypeCount = 5; // TileType.cs: Slippery through Gravel.

    [Serializable] public class Route
    {
        public float spawn_x, spawn_z, goal_x, goal_z, heading_deg;
    }

    [Serializable] public class Scenario
    {
        public int scenario_id, map_index;
        public string map_file;
        public Route[] routes;
    }

    [Serializable] public class ScenarioCollection
    {
        public Scenario[] scenarios;
    }

    public GridManager gridManager;
    [Tooltip("Pool of traffic cars in seat order. Add more cars in the Editor and rebuild to raise the maximum.")]
    public CarAgent[] cars;
    [Tooltip("Editor menu 'Expand open traffic scene to desired car pool' uses this value. " +
             "Fixed-route manifests need at least this many routes per scenario.")]
    [Min(2)] public int editorDesiredPoolSize = 4;
    [Min(2)] public int defaultActiveCarCount = 2;
    public TextAsset scenarioJson;
    [Min(0)] public int scenarioIndex = 0;
    public bool cycleScenarios = false;
    [Tooltip("Sample a map from the assigned manifest each episode; intended for training only.")]
    public bool randomizeScenarioEachEpisode = false;
    [Tooltip("Use held-out VoronoiVal maps. Disable only in the separate traffic training scene.")]
    public bool useValidationMaps = true;
    [Tooltip("Sample separated, reachable asphalt spawn/goal locations anew each episode. " +
             "The scenario JSON then selects maps only; fixed pilot routes remain unchanged.")]
    public bool randomizeRoutes = false;
    [Min(0.1f)] public float minEndpointSeparationM = 8f;
    [Min(0.1f)] public float minOwnSpawnGoalDistanceM = 20f;
    [Min(1)] public int placementAttempts = 50;
    [Min(1)] public int sharedMaxSteps = 5000;
    [Header("Optional smooth traffic curricula (training only)")]
    [Tooltip("Sample 2--10 active cars independently at each shared-episode reset. " +
             "The distribution centre stays low during warmup, ramps linearly, then stays high.")]
    public bool smoothCarCountCurriculum = false;
    [Min(2)] public int curriculumMinActiveCars = 2;
    [Min(2)] public int curriculumMaxActiveCars = 10;
    [Range(2f, 10f)] public float curriculumCarCountCentreStart = 2.5f;
    [Range(2f, 10f)] public float curriculumCarCountCentreEnd = 9.5f;
    [Min(0.1f)] public float carCountGaussianSigma = 1.4f;
    [Range(0f, 1f)] public float carCountUniformMix = 0.09f;
    [Tooltip("Optional ramp of the maximum straight-line spawn-to-goal distance. " +
             "Requires randomizeRoutes; zero means use the legacy wide-endpoint placement.")]
    public bool smoothRouteLengthCurriculum = false;
    [Min(0.1f)] public float curriculumRouteMinDistanceM = 0.1f;
    [Min(0.1f)] public float curriculumRouteDistanceStartM = 10f;
    [Min(0.1f)] public float curriculumRouteDistanceEndM = 70f;
    [Min(0f)] public float curriculumWarmupSteps = 5_000_000f;
    [Min(0f)] public float curriculumRampEndSteps = 35_000_000f;
    [Tooltip("Allow Python evaluation to choose a case with traffic_scenario_index.")]
    public bool useEnvironmentScenarioIndex = true;
    [Min(0f)] public float vehicleSensorInflationM = 0.1f;
    [Min(0.1f)] public float nearEncounterDistanceM = 10f;
    [InspectorName("Remove cars after any outcome")]
    [Tooltip("When enabled, every finished traffic car is removed from physics, laser hits, and rendering until the shared episode resets. This includes Goal, VehicleCollision, Terminated, and MaxStep.")]
    public bool removeCarsAfterGoal = false;

    [Header("Optional recovery curriculum (training only)")]
    [Tooltip("Use the hand-authored TrafficRecovery maps and staged traffic lessons for the first recoveryCurriculumEndSteps trainer steps. Default off: normal traffic training/evaluation is unchanged.")]
    public bool recoveryCurriculumEnabled = false;
    [Min(1f)] public float recoveryCurriculumEndSteps = 13_000_000f;
    [Tooltip("Keep a small amount of route-heading noise, so the policy cannot memorise exact spawn rotations.")]
    [Range(0f, 20f)] public float recoveryHeadingJitterDeg = 8f;
    [Header("Optional parked-vehicle recovery shaping")]
    [Tooltip("One-off reward after a car approaches the parked blocker then backs away far enough. Zero disables it, preserving the unshaped curriculum.")]
    [Min(0f)] public float recoveryBlockerClearanceReward = 0f;
    [Min(0.1f)] public float recoveryBlockerApproachDistanceM = 6f;
    [Min(0.1f)] public float recoveryBlockerBacktrackDistanceM = 3f;
    [Header("Editor recovery preview")]
    [Tooltip("Editor Play Mode only: ignore trainer step count and repeatedly run the selected recovery lesson. This is ignored in players/HPC builds.")]
    public bool forceRecoveryStageInEditor = false;
    [Tooltip("0 Straight; 1 S curve; 2 Turn; 3 Road mix; 4 Blind alley; 5 Dead-end detour; 6 Parked vehicle; 7 Two-car crossroads; 8 Passing bay; 9 Four-car crossroads; 10 head-to-head deadlock; 11 close-range circulation.")]
    [Range(0, 11)] public int editorRecoveryStageCode = 0;
    [Tooltip("Editor Play Mode only: sample a route difficulty within the selected lesson instead of always using its first, shortest route.")]
    public bool editorRandomizeRecoveryRouteDifficulty = true;
    [Range(0f, 1f)] public float editorRecoveryMinimumRouteProgress = 0.25f;

    private ScenarioCollection collection;
    private bool[] ready;
    private bool[] finished;
    private string[] outcomes;
    private BoxCollider[] bodyColliders;
    private float[,] activeTileSeconds;
    private int preparedEpisode = -1;
    private int episodeNumber;
    private int sharedStep;
    private bool running;
    private bool resetPending;
    private float minCarCenterDistanceM;
    private int nearEncounterSteps;
    private int activeCarCount;
    private Route[] episodeRoutes;
    private int placementSeed;
    private bool recordTrajectories;
    private int trajectoryStrideSteps;
    private int inspectorSharedMaxSteps;
    private int configuredNormalActiveCarCount;
    private TrafficVehicleObstacle parkedVehicle;
    private BoxCollider parkedVehicleCollider;
    private bool[] recoveryBlockerApproached;
    private bool[] recoveryBlockerClearanceAwarded;
    private Vector3[] recoveryBlockerSpawnPositions;
    private Vector3[] recoveryBlockerInitialForwards;
    private string recoveryStageName = "";
    private int recoveryStageCode = -1;
    // Pooled cars can invoke OnEpisodeBegin when their GameObject is enabled.
    // Suppress that callback while a recovery episode is being assembled, then
    // place the complete group together once all route data is valid.
    private bool preparingRecoveryGroup;

    public int SharedStep => sharedStep;
    public int SharedMaxSteps => sharedMaxSteps;
    public int ActiveCarCount => activeCarCount;
    public bool IsFinished(CarAgent car) => finished != null && finished[Seat(car)];

    void Awake()
    {
        if (scenarioJson == null || gridManager == null || cars == null || cars.Length < 2)
        {
            Debug.LogError("[Traffic] Assign a grid, a pool of cars and a scenario JSON.", this);
            enabled = false;
            return;
        }
        inspectorSharedMaxSteps = sharedMaxSteps;
        activeCarCount = defaultActiveCarCount;
        // ``traffic_remove_completed_goal_cars`` is retained as a backwards-
        // compatible YAML alias. New configurations write the clearer
        // ``traffic_remove_finished_cars`` key.
        float legacyRemoval = Academy.Instance.EnvironmentParameters.GetWithDefault(
            "traffic_remove_completed_goal_cars", removeCarsAfterGoal ? 1f : 0f);
        removeCarsAfterGoal = Academy.Instance.EnvironmentParameters.GetWithDefault(
            "traffic_remove_finished_cars", legacyRemoval) > 0.5f;
        smoothCarCountCurriculum = Academy.Instance.EnvironmentParameters.GetWithDefault(
            "traffic_smooth_car_count_curriculum_enabled", smoothCarCountCurriculum ? 1f : 0f) > 0.5f;
        smoothRouteLengthCurriculum = Academy.Instance.EnvironmentParameters.GetWithDefault(
            "traffic_smooth_route_length_curriculum_enabled", smoothRouteLengthCurriculum ? 1f : 0f) > 0.5f;
        recoveryCurriculumEnabled = Academy.Instance.EnvironmentParameters.GetWithDefault(
            "traffic_recovery_curriculum_enabled", recoveryCurriculumEnabled ? 1f : 0f) > 0.5f;
        // Keep this a local inspection aid. It must not be possible for a
        // checked Inspector value to alter a Linux player or HPC experiment.
        if (Application.isEditor && forceRecoveryStageInEditor)
            recoveryCurriculumEnabled = true;
        string[] commandLine = Environment.GetCommandLineArgs();
        for (int i = 0; i < commandLine.Length; i++)
        {
            if (commandLine[i] != "-traffic-car-count") continue;
            if (i + 1 >= commandLine.Length || !int.TryParse(commandLine[i + 1], out activeCarCount))
            {
                Debug.LogError("[Traffic] -traffic-car-count needs an integer value.", this);
                enabled = false;
                return;
            }
            break;
        }
        if (activeCarCount < 1 || activeCarCount > cars.Length)
        {
            Debug.LogError($"[Traffic] Requested {activeCarCount} cars; this build has a pool of " +
                           $"{cars.Length}. Choose 1..{cars.Length} or rebuild with a larger pool.", this);
            enabled = false;
            return;
        }
        configuredNormalActiveCarCount = activeCarCount;

        // A traffic manager only has useful shared-episode semantics when
        // there are at least two vehicles.  In particular, a one-seat shared
        // manager can leave the trainer with an indefinitely open episode in
        // headless multi-environment training.  Treat the command-line
        // one-car case as genuine solo pretraining instead: retain the same
        // GridManager and 89-observation car prefab, but hand episode resets
        // back to CarAgent's normal goal/terminal/MaxStep implementation.
        // Multi-agent runs (2, 5, and 10 cars) do not enter this branch.
        if (activeCarCount == 1 && !recoveryCurriculumEnabled)
        {
            ConfigureSoloMode();
            return;
        }
        collection = JsonUtility.FromJson<ScenarioCollection>(scenarioJson.text);
        if (collection?.scenarios == null || collection.scenarios.Length == 0)
        {
            Debug.LogError("[Traffic] Scenario JSON has no scenarios.", this);
            enabled = false;
            return;
        }
        // Keep shared-state arrays at the pool size.  An optional curriculum
        // can change only the active prefix between episodes without changing
        // the pool or Unity behaviour specification.
        ready = new bool[cars.Length];
        finished = new bool[cars.Length];
        outcomes = new string[cars.Length];
        bodyColliders = new BoxCollider[cars.Length];
        activeTileSeconds = new float[cars.Length, TileTypeCount];
        recoveryBlockerApproached = new bool[cars.Length];
        recoveryBlockerClearanceAwarded = new bool[cars.Length];
        recoveryBlockerSpawnPositions = new Vector3[cars.Length];
        recoveryBlockerInitialForwards = new Vector3[cars.Length];
        for (int i = 0; i < cars.Length; i++)
        {
            if (cars[i] == null) throw new InvalidOperationException("Traffic car is missing at seat " + i);
            cars[i].trafficManager = this;
            cars[i].trafficSeatIndex = i;
            cars[i].MaxStep = 0; // One clock for all cars; no individual auto-reset.
            bodyColliders[i] = cars[i].GetComponent<BoxCollider>();
            if (bodyColliders[i] == null)
                throw new InvalidOperationException("Traffic car needs a body BoxCollider at seat " + i);
        }
        CreateParkedVehicle();
        // Extra cars are pooled in the build but have no Agent behavior/physics until selected.
        for (int i = 0; i < cars.Length; i++)
        {
            bool active = i < activeCarCount;
            cars[i].gameObject.SetActive(active);
            cars[i].goal.gameObject.SetActive(active);
            cars[i].spawnPoint.gameObject.SetActive(active);
        }
        Debug.Log($"[Traffic] Active cars: {activeCarCount}/{cars.Length}; " +
                  $"scenario count: {collection.scenarios.Length}; remove finished cars: {removeCarsAfterGoal}; " +
                  $"recovery curriculum: {recoveryCurriculumEnabled}.");

#if UNITY_EDITOR
        // Trails and per-seat goal colours are an Editor play-mode viewing aid.
        // They are intentionally not added to Linux training/evaluation players.
        if (GetComponent<TrafficCarTrails>() == null)
            gameObject.AddComponent<TrafficCarTrails>();
#endif
    }

    void ConfigureSoloMode()
    {
        for (int i = 0; i < cars.Length; i++)
        {
            bool active = i == 0;
            // CarAgent checks this reference to select all traffic-specific
            // endpoint, collision, and shared-clock handling.  Clearing it
            // restores the standard one-car map randomisation and its YAML
            // max_step_budget (5,000 in the current configuration).
            cars[i].trafficManager = null;
            cars[i].trafficSeatIndex = -1;
            cars[i].gameObject.SetActive(active);
            cars[i].goal.gameObject.SetActive(active);
            cars[i].spawnPoint.gameObject.SetActive(active);
        }
        Debug.Log("[Traffic] One-car command-line mode: TrafficScenarioManager disabled; " +
                  "using normal CarAgent episodes and map placement.");
        enabled = false;
    }

    int Seat(CarAgent car)
    {
        int seat = car.trafficSeatIndex;
        if (seat < 0 || seat >= activeCarCount || cars[seat] != car)
            throw new InvalidOperationException("Car is not registered with this traffic manager.");
        return seat;
    }

    Scenario SelectedScenario()
    {
        int index = randomizeScenarioEachEpisode
            ? UnityEngine.Random.Range(0, collection.scenarios.Length)
            : cycleScenarios ? (scenarioIndex + episodeNumber) % collection.scenarios.Length : scenarioIndex;
        if (useEnvironmentScenarioIndex)
        {
            float overrideIndex = Academy.Instance.EnvironmentParameters.GetWithDefault("traffic_scenario_index", -1f);
            if (overrideIndex >= 0f) index = Mathf.RoundToInt(overrideIndex);
        }
        if (index < 0 || index >= collection.scenarios.Length)
            throw new InvalidOperationException($"Traffic scenario {index} is outside 0.." +
                                                $"{collection.scenarios.Length - 1}.");
        return collection.scenarios[index];
    }

    void PrepareEpisode()
    {
        RefreshCurriculumSettings();
        bool useRecovery = IsRecoveryEpisode();
        Scenario scenario = null;
        int recoveryActiveCarCount = -1;
        if (useRecovery)
        {
            preparingRecoveryGroup = true;
            recoveryActiveCarCount = PrepareRecoveryEpisode();
        }
        else
        {
            recoveryStageName = "";
            recoveryStageCode = -1;
            DisableParkedVehicle();
            if (smoothCarCountCurriculum)
                SetActiveCarCount(SampleCurriculumCarCount());
            else if (recoveryCurriculumEnabled)
                // A 6M recovery run may end its last lesson with four cars.
                // Resume the command-line/Inspector traffic count (normally 10)
                // before returning to ordinary random-map training.
                SetActiveCarCount(configuredNormalActiveCarCount);
            scenario = SelectedScenario();
            if (!randomizeRoutes && (scenario.routes == null || scenario.routes.Length < activeCarCount))
                throw new InvalidOperationException("Traffic scenario does not have a route for every car.");
            gridManager.forceAllNormal = false;
            gridManager.useRecoveryMaps = false;
            gridManager.useValidationMaps = useValidationMaps;
            gridManager.forcedMapIndex = scenario.map_index;
            gridManager.Regenerate();
            if (gridManager.LastLoadedVoronoiMapIndex != scenario.map_index ||
                gridManager.LastLoadedVoronoiMapName != Path.GetFileNameWithoutExtension(scenario.map_file))
                throw new InvalidOperationException($"Traffic scenario {scenario.scenario_id} expected map " +
                                                    $"{scenario.map_index} ({scenario.map_file}), but " +
                                                    $"GridManager loaded {gridManager.LastLoadedVoronoiMapIndex} " +
                                                    $"({gridManager.LastLoadedVoronoiMapName}).");
            if (randomizeRoutes)
            {
                float requestedSeed = Academy.Instance.EnvironmentParameters.GetWithDefault(
                    "traffic_placement_seed", -1f);
                // Environment parameters are floats; keep evaluator-provided seeds below 2^24.
                int baseSeed = requestedSeed >= 0f
                    ? Mathf.RoundToInt(requestedSeed)
                    : UnityEngine.Random.Range(0, 10000000);
                placementSeed = unchecked(baseSeed + episodeNumber * 100003);
                episodeRoutes = SampleRoutes(new System.Random(placementSeed));
            }
            else
            {
                placementSeed = -1;
                episodeRoutes = scenario.routes;
            }
        }
        Array.Clear(ready, 0, ready.Length);
        Array.Clear(finished, 0, finished.Length);
        Array.Clear(outcomes, 0, outcomes.Length);
        Array.Clear(activeTileSeconds, 0, activeTileSeconds.Length);
        sharedStep = 0;
        minCarCenterDistanceM = float.PositiveInfinity;
        nearEncounterSteps = 0;
        recordTrajectories = Academy.Instance.EnvironmentParameters.GetWithDefault(
            "traffic_record_trajectories", 0f) > 0.5f;
        trajectoryStrideSteps = Mathf.Max(1, Mathf.RoundToInt(
            Academy.Instance.EnvironmentParameters.GetWithDefault("traffic_trajectory_stride_steps", 5f)));
        running = false;
        resetPending = false;
        preparedEpisode = episodeNumber;
        if (recoveryActiveCarCount > 0)
        {
            // Activating the extra pooled cars now is safe: the map, routes,
            // and reset arrays already belong to this episode.
            SetActiveCarCount(recoveryActiveCarCount);
            PlacePreparedRecoveryGroup();
            preparingRecoveryGroup = false;
        }
        string episodeLabel = useRecovery
            ? $"recovery {recoveryStageName}"
            : $"scenario {scenario.scenario_id}, map {scenario.map_file}";
        Debug.Log($"[Traffic] Episode {episodeNumber}, {episodeLabel}, cars {activeCarCount}, " +
                  $"placement seed {placementSeed}.");
        Academy.Instance.StatsRecorder.Add("Custom/TrafficCurriculumCarCountCenter",
            CurrentCarCountCentre(), StatAggregationMethod.MostRecent);
        float routeMax = CurrentRouteMaxDistanceM();
        Academy.Instance.StatsRecorder.Add("Custom/TrafficCurriculumRouteMaxDistanceM",
            float.IsPositiveInfinity(routeMax) ? -1f : routeMax, StatAggregationMethod.MostRecent);
        Academy.Instance.StatsRecorder.Add("Custom/TrafficRecoveryActive", useRecovery ? 1f : 0f,
            StatAggregationMethod.MostRecent);
        Academy.Instance.StatsRecorder.Add("Custom/TrafficRecoveryStage", recoveryStageCode,
            StatAggregationMethod.MostRecent);
        Academy.Instance.StatsRecorder.Add("Custom/TrafficCurriculumGlobalStep", TrainingSteps(),
            StatAggregationMethod.MostRecent);
    }

    bool IsRecoveryEpisode()
    {
        return (Application.isEditor && forceRecoveryStageInEditor) ||
               (recoveryCurriculumEnabled && TrainingSteps() < recoveryCurriculumEndSteps);
    }

    enum RecoveryStage
    {
        Straight, SCurve, Turn, RoadMix, BlindAlley, DeadEndDetour,
        ParkedVehicle, CrossroadsTwo, PassingBay, CrossroadsFour,
        CloseRangeDeadlock, CloseRangeCirculation
    }

    RecoveryStage CurrentRecoveryStage()
    {
        if (Application.isEditor && forceRecoveryStageInEditor)
            return (RecoveryStage)Mathf.Clamp(editorRecoveryStageCode, 0, (int)RecoveryStage.CloseRangeCirculation);
        float step = TrainingSteps();
        if (step < 200_000f) return RecoveryStage.Straight;
        if (step < 600_000f) return RecoveryStage.SCurve;
        if (step < 1_000_000f) return RecoveryStage.Turn;
        if (step < 3_000_000f) return RecoveryStage.RoadMix;
        // Stage 5 in the written curriculum is the blind-alley reverse
        // lesson. Give its new near-to-far goal progression 50% more time.
        if (step < 4_500_000f) return RecoveryStage.BlindAlley;
        if (step < 5_000_000f) return RecoveryStage.DeadEndDetour;
        if (step < 7_000_000f) return RecoveryStage.ParkedVehicle;
        if (step < 8_000_000f) return RecoveryStage.CrossroadsTwo;
        if (step < 8_500_000f) return RecoveryStage.PassingBay;
        if (step < 10_000_000f) return RecoveryStage.CrossroadsFour;
        if (step < 12_000_000f) return RecoveryStage.CloseRangeDeadlock;
        return RecoveryStage.CloseRangeCirculation;
    }

    int PrepareRecoveryEpisode()
    {
        RecoveryStage stage = CurrentRecoveryStage();
        recoveryStageName = stage.ToString();
        recoveryStageCode = (int)stage;
        int desiredCars = (stage == RecoveryStage.CrossroadsFour || stage == RecoveryStage.CloseRangeDeadlock ||
                           stage == RecoveryStage.CloseRangeCirculation) ? 4 :
                          (stage == RecoveryStage.CrossroadsTwo || stage == RecoveryStage.PassingBay) ? 2 : 1;

        float requestedSeed = Academy.Instance.EnvironmentParameters.GetWithDefault(
            "traffic_placement_seed", -1f);
        int baseSeed = requestedSeed >= 0f ? Mathf.RoundToInt(requestedSeed) : UnityEngine.Random.Range(0, 10_000_000);
        placementSeed = unchecked(baseSeed + episodeNumber * 100_003);
        var rng = new System.Random(placementSeed);

        string mapName = RecoveryMapForStage(stage, rng);
        int mapIndex = gridManager.RecoveryMapIndex(mapName);
        if (mapIndex < 0)
            throw new InvalidOperationException($"Recovery map '{mapName}' was not found in Resources/Maps/TrafficRecovery.");
        gridManager.forceAllNormal = false;
        gridManager.useValidationMaps = false;
        gridManager.useRecoveryMaps = true;
        gridManager.forcedMapIndex = mapIndex;
        gridManager.Regenerate();
        if (gridManager.LastLoadedVoronoiMapName != Path.GetFileNameWithoutExtension(mapName))
            throw new InvalidOperationException($"Recovery requested {mapName}, but GridManager loaded " +
                                                $"{gridManager.LastLoadedVoronoiMapName}.");

        DisableParkedVehicle();
        episodeRoutes = RecoveryRoutes(stage, rng);
        ConfigureRecoveryBlockerClearance(stage, desiredCars);
        return desiredCars;
    }

    void ConfigureRecoveryBlockerClearance(RecoveryStage stage, int desiredCars)
    {
        Array.Clear(recoveryBlockerApproached, 0, recoveryBlockerApproached.Length);
        Array.Clear(recoveryBlockerClearanceAwarded, 0, recoveryBlockerClearanceAwarded.Length);
        if (stage != RecoveryStage.ParkedVehicle || parkedVehicle == null ||
            !parkedVehicle.gameObject.activeInHierarchy)
            return;

        for (int seat = 0; seat < desiredCars; seat++)
        {
            Route route = episodeRoutes[seat];
            recoveryBlockerSpawnPositions[seat] = new Vector3(route.spawn_x, 0f, route.spawn_z);
            recoveryBlockerInitialForwards[seat] = Quaternion.Euler(0f, route.heading_deg, 0f) * Vector3.forward;
        }
    }

    // Called by CarAgent once per decision. This is deliberately a one-time
    // recovery-only milestone: it rewards backing out after genuinely reaching
    // the stationary blocker, not continuous reverse driving or simply taking
    // a long route around it. A zero configured reward leaves the curriculum
    // exactly unshaped, enabling a matched A/B training comparison.
    public float ConsumeRecoveryBlockerClearanceReward(CarAgent car)
    {
        if (recoveryBlockerClearanceReward <= 0f || recoveryStageCode != (int)RecoveryStage.ParkedVehicle ||
            parkedVehicle == null || !parkedVehicle.gameObject.activeInHierarchy)
            return 0f;
        int seat = Seat(car);
        if (recoveryBlockerClearanceAwarded[seat]) return 0f;

        float blockerDistance = Vector3.Distance(car.transform.position, parkedVehicle.transform.position);
        if (blockerDistance <= recoveryBlockerApproachDistanceM)
            recoveryBlockerApproached[seat] = true;
        if (!recoveryBlockerApproached[seat]) return 0f;

        Vector3 displacement = car.transform.position - recoveryBlockerSpawnPositions[seat];
        float signedForwardDisplacement = Vector3.Dot(displacement, recoveryBlockerInitialForwards[seat]);
        if (signedForwardDisplacement > -recoveryBlockerBacktrackDistanceM ||
            blockerDistance < recoveryBlockerApproachDistanceM)
            return 0f;

        recoveryBlockerClearanceAwarded[seat] = true;
        Academy.Instance.StatsRecorder.Add("Custom/RecoveryBlockerClearanceAchieved", 1f, StatAggregationMethod.Sum);
        Academy.Instance.StatsRecorder.Add("Custom/RecoveryBlockerClearanceReward", recoveryBlockerClearanceReward,
            StatAggregationMethod.Sum);
        return recoveryBlockerClearanceReward;
    }

    void PlacePreparedRecoveryGroup()
    {
        if (episodeRoutes == null || episodeRoutes.Length < activeCarCount)
            throw new InvalidOperationException("Recovery episode does not have a route for every active car.");
        for (int seat = 0; seat < activeCarCount; seat++)
        {
            cars[seat].PlaceForTraffic(episodeRoutes[seat]);
            ready[seat] = true;
        }
        running = true;
        Physics.SyncTransforms();
        RecordTrajectoryPositions(true);
    }

    string RecoveryMapForStage(RecoveryStage stage, System.Random rng)
    {
        switch (stage)
        {
            case RecoveryStage.Straight: return "recovery_01_straight";
            case RecoveryStage.SCurve: return "recovery_07_s_curve";
            case RecoveryStage.Turn: return "recovery_02_turn";
            case RecoveryStage.RoadMix:
                string[] roads = { "recovery_01_straight", "recovery_02_turn", "recovery_07_s_curve" };
                return roads[rng.Next(roads.Length)];
            case RecoveryStage.BlindAlley: return "recovery_06_blind_alley";
            case RecoveryStage.DeadEndDetour: return "recovery_08_dead_end_detour";
            case RecoveryStage.ParkedVehicle: return "recovery_05_passing_bay";
            case RecoveryStage.CrossroadsTwo:
            case RecoveryStage.CrossroadsFour:
            case RecoveryStage.CloseRangeDeadlock: return "recovery_04_crossroads";
            case RecoveryStage.CloseRangeCirculation: return "recovery_09_circulation_loop";
            case RecoveryStage.PassingBay: return "recovery_05_passing_bay";
            default: throw new ArgumentOutOfRangeException(nameof(stage));
        }
    }

    Route[] RecoveryRoutes(RecoveryStage stage, System.Random rng)
    {
        switch (stage)
        {
            case RecoveryStage.Straight: return new[] { StraightRoute(rng) };
            case RecoveryStage.SCurve: return new[] { SCurveRoute(rng) };
            case RecoveryStage.Turn: return new[] { TurnRoute(rng) };
            case RecoveryStage.RoadMix:
                int choice = rng.Next(3);
                return new[] { choice == 0 ? StraightRoute(rng) : choice == 1 ? TurnRoute(rng) : SCurveRoute(rng) };
            case RecoveryStage.BlindAlley:
                // Simple reverse lesson: the car begins near the capped end
                // facing Terminal, while its goal is back along the main road.
                // Across this 1.5M lesson the start moves deeper into the cap
                // and the goal moves farther down the main road.
                float blindProgress = RecoveryRouteProgress(3_000_000f, 1_500_000f, rng);
                int blindSpawnX = Mathf.Clamp(Mathf.RoundToInt(Mathf.Lerp(32f, 44f, blindProgress) +
                    ((float)rng.NextDouble() * 2f - 1f)), 32, 44);
                int blindGoalZ = Mathf.Clamp(Mathf.RoundToInt(Mathf.Lerp(35f, 10f, blindProgress) +
                    ((float)rng.NextDouble() * 4f - 2f)), 8, 37);
                return new[] { RouteCellsWithInitialDirection(
                    new Vector2Int(blindSpawnX, 45), new Vector2Int(18, blindGoalZ), Vector2.right, rng) };
            case RecoveryStage.DeadEndDetour:
                // The goal is spatially forward/right, but the direct upper
                // branch is capped by a terminal gap. Back out, use the lower
                // bypass, then approach the goal from below.
                return new[] { RouteCellsWithInitialDirection(
                    new Vector2Int(rng.Next(35, 40), 45),
                    new Vector2Int(50, rng.Next(34, 39)), Vector2.right, rng) };
            case RecoveryStage.ParkedVehicle:
                // Keep the blocked manoeuvre unambiguous first; introduce a
                // small easy-case mixture only in the final fifth so that the
                // learner does not equate every vehicle detection with reverse.
                bool easy = TrainingSteps() >= 6_600_000f && rng.NextDouble() < 0.20;
                int blockerZ = rng.Next(32, 35);
                if (!easy) EnableParkedVehicle(CellCenter(new Vector2Int(33, blockerZ)), 180f);
                return new[] { RouteCells(new Vector2Int(33, rng.Next(36, 40)),
                    new Vector2Int(33, rng.Next(8, 13)), rng) };
            case RecoveryStage.CrossroadsTwo:
                return CrossroadsRoutes(rng, 2);
            case RecoveryStage.PassingBay:
                return new[] {
                    RouteCells(new Vector2Int(33, rng.Next(8, 16)), new Vector2Int(33, rng.Next(51, 58)), rng),
                    RouteCells(new Vector2Int(33, rng.Next(50, 58)), new Vector2Int(33, rng.Next(8, 16)), rng),
                };
            case RecoveryStage.CrossroadsFour:
                return CrossroadsRoutes(rng, 4);
            case RecoveryStage.CloseRangeDeadlock:
                return CloseRangeDeadlockRoutes(rng);
            case RecoveryStage.CloseRangeCirculation:
                return CloseRangeCirculationRoutes(rng);
            default: throw new ArgumentOutOfRangeException(nameof(stage));
        }
    }

    Route StraightRoute(System.Random rng)
    {
        float progress = RecoveryRouteProgress(0f, 200_000f, rng);
        int distance = Mathf.RoundToInt(Mathf.Lerp(3f, 30f, progress));
        int startZ = rng.Next(8, Mathf.Max(9, 58 - distance));
        Vector2Int a = new Vector2Int(rng.Next(32, 35), startZ);
        Vector2Int b = new Vector2Int(a.x, startZ + distance);
        if (rng.Next(2) == 0) { Vector2Int swap = a; a = b; b = swap; }
        return RouteCells(a, b, rng);
    }

    Route TurnRoute(System.Random rng)
    {
        // Begin on a straight approach. As this 400k lesson progresses, the
        // sampled goal travels through the bend and then along its exit road.
        float progress = RecoveryRouteProgress(600_000f, 400_000f, rng);
        float distance = Mathf.Lerp(6f, 58f, progress) * Mathf.Lerp(.82f, 1f, (float)rng.NextDouble());
        return RouteAlongPath(new[] {
            new Vector2(18f, 8f), new Vector2(18f, 39f), new Vector2(49f, 39f)
        }, distance, rng);
    }

    Route SCurveRoute(System.Random rng)
    {
        // Sample the same interpolating spline used to generate the map.
        // Short routes remain nearly straight; longer routes reveal one bend,
        // then the full S. The tangent, rather than the far-away goal vector,
        // sets the initial road-aligned heading.
        float progress = RecoveryRouteProgress(200_000f, 400_000f, rng);
        float distance = Mathf.Lerp(7f, 78f, progress) * Mathf.Lerp(.82f, 1f, (float)rng.NextDouble());
        return RouteAlongPath(new[] {
            new Vector2(20.00f, 7.00f), new Vector2(18.23f, 14.01f),
            new Vector2(20.08f, 18.07f), new Vector2(24.48f, 20.12f),
            new Vector2(30.36f, 21.12f), new Vector2(36.66f, 22.02f),
            new Vector2(42.30f, 23.79f), new Vector2(46.22f, 27.37f),
            new Vector2(47.57f, 33.00f), new Vector2(46.22f, 38.63f),
            new Vector2(42.30f, 42.21f), new Vector2(36.66f, 43.98f),
            new Vector2(30.36f, 44.88f), new Vector2(24.48f, 45.88f),
            new Vector2(20.08f, 47.93f), new Vector2(18.23f, 51.99f),
            new Vector2(20.00f, 59.00f),
        }, distance, rng);
    }

    float RecoveryRouteProgress(float lessonStartStep, float lessonDurationSteps, System.Random rng)
    {
        // The trainer deliberately moves each early driving lesson from a
        // short, easy route to a long route. In the Editor there is no trainer
        // global step, so preview a useful random difficulty instead of always
        // showing the shortest route at progress zero.
        if (Application.isEditor && forceRecoveryStageInEditor && editorRandomizeRecoveryRouteDifficulty)
        {
            float minimum = Mathf.Clamp01(editorRecoveryMinimumRouteProgress);
            return Mathf.Lerp(minimum, 1f, (float)rng.NextDouble());
        }
        return Mathf.Clamp01((TrainingSteps() - lessonStartStep) / lessonDurationSteps);
    }

    Route RouteAlongPath(Vector2[] points, float requestedDistanceCells, System.Random rng)
    {
        if (points == null || points.Length < 2)
            throw new ArgumentException("A recovery route needs at least two centre-line points.");
        var path = (Vector2[])points.Clone();
        if (rng.Next(2) == 0) Array.Reverse(path);
        float total = 0f;
        for (int i = 1; i < path.Length; i++) total += Vector2.Distance(path[i - 1], path[i]);
        float distance = Mathf.Clamp(requestedDistanceCells, 2f, total - .5f);
        float freeStartDistance = Mathf.Max(0f, total - distance);
        float startDistance = freeStartDistance <= .01f ? 0f :
            (float)rng.NextDouble() * freeStartDistance;
        Vector2 startPoint = PointOnPath(path, startDistance, out Vector2 tangent);
        Vector2 goalPoint = PointOnPath(path, startDistance + distance, out _);
        Vector3 start = CellCenter(startPoint);
        Vector3 goal = CellCenter(goalPoint);
        float heading = Mathf.Atan2(tangent.x, tangent.y) * Mathf.Rad2Deg;
        heading += ((float)rng.NextDouble() * 2f - 1f) * recoveryHeadingJitterDeg;
        return new Route { spawn_x = start.x, spawn_z = start.z,
            goal_x = goal.x, goal_z = goal.z, heading_deg = heading };
    }

    Vector2 PointOnPath(Vector2[] path, float distanceFromStart, out Vector2 tangent)
    {
        float left = Mathf.Max(0f, distanceFromStart);
        for (int i = 1; i < path.Length; i++)
        {
            Vector2 segment = path[i] - path[i - 1];
            float length = segment.magnitude;
            if (left <= length)
            {
                tangent = segment.normalized;
                return path[i - 1] + segment * (left / length);
            }
            left -= length;
        }
        Vector2 finalSegment = path[path.Length - 1] - path[path.Length - 2];
        tangent = finalSegment.normalized;
        return path[path.Length - 1];
    }

    Route[] CrossroadsRoutes(System.Random rng, int count)
    {
        // Small independently sampled offsets break the otherwise permanent symmetry of a shared policy.
        var routes = new List<Route> {
            RouteCells(new Vector2Int(rng.Next(32, 35), rng.Next(7, 16)), new Vector2Int(rng.Next(32, 35), rng.Next(51, 59)), rng),
            RouteCells(new Vector2Int(rng.Next(32, 35), rng.Next(51, 59)), new Vector2Int(rng.Next(32, 35), rng.Next(7, 16)), rng),
            RouteCells(new Vector2Int(rng.Next(7, 16), rng.Next(32, 35)), new Vector2Int(rng.Next(51, 59), rng.Next(32, 35)), rng),
            RouteCells(new Vector2Int(rng.Next(51, 59), rng.Next(32, 35)), new Vector2Int(rng.Next(7, 16), rng.Next(32, 35)), rng),
        };
        return routes.GetRange(0, count).ToArray();
    }

    Route[] CloseRangeDeadlockRoutes(System.Random rng)
    {
        // Four cars begin only a few road cells from the intersection, each
        // facing a car and a goal directly beyond it. This is deliberately a
        // close, symmetric deadlock: proceeding straight causes a collision,
        // while reversing, yielding, and then re-entering the junction can
        // resolve it. Small lateral offsets and heading jitter prevent one
        // exact layout from being memorised.
        int verticalX = rng.Next(32, 35);
        int horizontalZ = rng.Next(32, 35);
        int near = rng.Next(28, 31);
        int far = rng.Next(36, 39);
        return new[] {
            RouteCells(new Vector2Int(verticalX, near), new Vector2Int(verticalX, far), rng),
            RouteCells(new Vector2Int(verticalX, far), new Vector2Int(verticalX, near), rng),
            RouteCells(new Vector2Int(near, horizontalZ), new Vector2Int(far, horizontalZ), rng),
            RouteCells(new Vector2Int(far, horizontalZ), new Vector2Int(near, horizontalZ), rng),
        };
    }

    Route[] CloseRangeCirculationRoutes(System.Random rng)
    {
        // Four nearby cars form the cyclic arrangement in the recovery
        // diagram, but on an open asphalt plaza rather than a road loop.
        // Each nose is aimed 1.5 m from the centre of the next car's side,
        // at a safe 4.74 m centre-to-centre distance. This is close enough
        // to require an immediate yield, but avoids physics overlap at reset.
        // Goals remain directly ahead on clear asphalt, isolating reaction
        // to other vehicles from road-edge following.
        int jx = rng.Next(-1, 2);
        int jz = rng.Next(-1, 2);
        return new[] {
            RouteCellsWithInitialDirection(new Vector2Int(32 + jx, 34 + jz),
                new Vector2Int(45 + jx, 34 + jz), Vector2.right, rng),
            RouteCellsWithInitialDirection(new Vector2Int(35 + jx, 33 + jz),
                new Vector2Int(35 + jx, 20 + jz), Vector2.down, rng),
            RouteCellsWithInitialDirection(new Vector2Int(34 + jx, 30 + jz),
                new Vector2Int(20 + jx, 30 + jz), Vector2.left, rng),
            RouteCellsWithInitialDirection(new Vector2Int(31 + jx, 31 + jz),
                new Vector2Int(31 + jx, 45 + jz), Vector2.up, rng),
        };
    }

    Route RouteCells(Vector2Int spawn, Vector2Int goal, System.Random rng)
    {
        Vector3 start = CellCenter(spawn);
        Vector3 end = CellCenter(goal);
        Vector3 direction = end - start;
        float heading = Mathf.Atan2(direction.x, direction.z) * Mathf.Rad2Deg;
        heading += ((float)rng.NextDouble() * 2f - 1f) * recoveryHeadingJitterDeg;
        return new Route { spawn_x = start.x, spawn_z = start.z, goal_x = end.x, goal_z = end.z, heading_deg = heading };
    }

    Route RouteCellsWithInitialDirection(Vector2Int spawn, Vector2Int goal,
        Vector2 initialDirection, System.Random rng)
    {
        Vector3 start = CellCenter(spawn);
        Vector3 end = CellCenter(goal);
        float heading = Mathf.Atan2(initialDirection.x, initialDirection.y) * Mathf.Rad2Deg;
        heading += ((float)rng.NextDouble() * 2f - 1f) * recoveryHeadingJitterDeg;
        return new Route { spawn_x = start.x, spawn_z = start.z, goal_x = end.x, goal_z = end.z, heading_deg = heading };
    }

    Vector3 CellCenter(Vector2 cell)
    {
        float sizeM = gridManager.gridSize * gridManager.cellSize;
        return new Vector3(-sizeM * 0.5f + (cell.x + 0.5f) * gridManager.cellSize,
                           0f,
                           -sizeM * 0.5f + (cell.y + 0.5f) * gridManager.cellSize);
    }

    // During an ML-Agents run the trainer patch publishes its authoritative
    // global decision count. Unlike Academy.TotalStepCount this remains in
    // exactly the same units as PPO max_steps when the curriculum changes the
    // number of active cars. The fallback preserves the old local-editor and
    // unpatched-trainer behaviour.
    float TrainingSteps()
    {
        float trainerGlobalStep = Academy.Instance.EnvironmentParameters.GetWithDefault(
            "traffic_recovery_global_step", -1f);
        if (trainerGlobalStep >= 0f) return trainerGlobalStep;
        float environments = Mathf.Max(1f, Academy.Instance.EnvironmentParameters.GetWithDefault(
            "training_num_envs", 1f));
        var requester = cars != null && cars.Length > 0 ? cars[0].GetComponent<DecisionRequester>() : null;
        float decisionPeriod = requester != null ? Mathf.Max(1, requester.DecisionPeriod) : 1f;
        return Academy.Instance.TotalStepCount * environments / decisionPeriod;
    }

    float CurriculumProgress()
    {
        float start = Mathf.Max(0f, curriculumWarmupSteps);
        float end = Mathf.Max(start, curriculumRampEndSteps);
        if (TrainingSteps() <= start) return 0f;
        if (TrainingSteps() >= end || end <= start) return 1f;
        return (TrainingSteps() - start) / (end - start);
    }

    void RefreshCurriculumSettings()
    {
        var ep = Academy.Instance.EnvironmentParameters;
        // Evaluation can extend the common traffic clock without changing the
        // training YAML or the Inspector default.  A missing/negative value
        // deliberately restores the build's normal 5,000-step budget.
        float requestedSharedMaxSteps = ep.GetWithDefault("traffic_shared_max_steps", -1f);
        sharedMaxSteps = requestedSharedMaxSteps >= 1f
            ? Mathf.RoundToInt(requestedSharedMaxSteps)
            : inspectorSharedMaxSteps;
        smoothCarCountCurriculum = ep.GetWithDefault("traffic_smooth_car_count_curriculum_enabled",
            smoothCarCountCurriculum ? 1f : 0f) > 0.5f;
        smoothRouteLengthCurriculum = ep.GetWithDefault("traffic_smooth_route_length_curriculum_enabled",
            smoothRouteLengthCurriculum ? 1f : 0f) > 0.5f;
        recoveryCurriculumEnabled = ep.GetWithDefault("traffic_recovery_curriculum_enabled",
            recoveryCurriculumEnabled ? 1f : 0f) > 0.5f;
        recoveryCurriculumEndSteps = Mathf.Max(1f, ep.GetWithDefault(
            "traffic_recovery_curriculum_end_steps", recoveryCurriculumEndSteps));
        recoveryBlockerClearanceReward = Mathf.Max(0f, ep.GetWithDefault(
            "traffic_recovery_blocker_clearance_reward", recoveryBlockerClearanceReward));
        curriculumMinActiveCars = Mathf.Clamp(Mathf.RoundToInt(ep.GetWithDefault(
            "traffic_car_count_min", curriculumMinActiveCars)), 2, cars.Length);
        curriculumMaxActiveCars = Mathf.Clamp(Mathf.RoundToInt(ep.GetWithDefault(
            "traffic_car_count_max", curriculumMaxActiveCars)), curriculumMinActiveCars, cars.Length);
        carCountGaussianSigma = Mathf.Max(0.1f, ep.GetWithDefault(
            "traffic_car_count_gaussian_sigma", carCountGaussianSigma));
        carCountUniformMix = Mathf.Clamp01(ep.GetWithDefault(
            "traffic_car_count_uniform_mix", carCountUniformMix));
        curriculumCarCountCentreStart = Mathf.Clamp(ep.GetWithDefault(
            "traffic_car_count_centre_start", curriculumCarCountCentreStart), curriculumMinActiveCars, curriculumMaxActiveCars);
        curriculumCarCountCentreEnd = Mathf.Clamp(ep.GetWithDefault(
            "traffic_car_count_centre_end", curriculumCarCountCentreEnd), curriculumMinActiveCars, curriculumMaxActiveCars);
        curriculumRouteMinDistanceM = Mathf.Max(0.1f, ep.GetWithDefault(
            "traffic_route_distance_min_m", curriculumRouteMinDistanceM));
        curriculumRouteDistanceStartM = Mathf.Max(curriculumRouteMinDistanceM, ep.GetWithDefault(
            "traffic_route_distance_start_m", curriculumRouteDistanceStartM));
        curriculumRouteDistanceEndM = Mathf.Max(curriculumRouteMinDistanceM, ep.GetWithDefault(
            "traffic_route_distance_end_m", curriculumRouteDistanceEndM));
        curriculumWarmupSteps = Mathf.Max(0f, ep.GetWithDefault(
            "traffic_curriculum_warmup_steps", curriculumWarmupSteps));
        curriculumRampEndSteps = Mathf.Max(curriculumWarmupSteps, ep.GetWithDefault(
            "traffic_curriculum_ramp_end_steps", curriculumRampEndSteps));
    }

    float CurrentCarCountCentre()
    {
        return smoothCarCountCurriculum
            ? Mathf.Lerp(curriculumCarCountCentreStart, curriculumCarCountCentreEnd, CurriculumProgress())
            : activeCarCount;
    }

    float CurrentRouteMaxDistanceM()
    {
        if (!smoothRouteLengthCurriculum) return 0f;
        // The final five million steps deliberately have no upper limit: the
        // policy finishes on the ordinary full-route distribution rather than
        // spending the end of training capped at an artificial distance.
        if (TrainingSteps() >= curriculumRampEndSteps) return float.PositiveInfinity;
        return Mathf.Lerp(curriculumRouteDistanceStartM, curriculumRouteDistanceEndM, CurriculumProgress());
    }

    int SampleCurriculumCarCount()
    {
        float centre = CurrentCarCountCentre();
        float[] weights = new float[curriculumMaxActiveCars - curriculumMinActiveCars + 1];
        float total = 0f;
        for (int i = 0; i < weights.Length; i++)
        {
            int count = curriculumMinActiveCars + i;
            float gaussian = Mathf.Exp(-0.5f * Mathf.Pow((count - centre) / carCountGaussianSigma, 2f));
            weights[i] = (1f - carCountUniformMix) * gaussian + carCountUniformMix / weights.Length;
            total += weights[i];
        }
        float draw = UnityEngine.Random.value * total;
        for (int i = 0; i < weights.Length; i++)
        {
            draw -= weights[i];
            if (draw <= 0f) return curriculumMinActiveCars + i;
        }
        return curriculumMaxActiveCars;
    }

    void SetActiveCarCount(int nextCount)
    {
        nextCount = Mathf.Clamp(nextCount, recoveryCurriculumEnabled ? 1 : 2, cars.Length);
        if (nextCount == activeCarCount) return;
        activeCarCount = nextCount;
        for (int i = 0; i < cars.Length; i++)
        {
            bool active = i < activeCarCount;
            if (cars[i].gameObject.activeSelf != active) cars[i].gameObject.SetActive(active);
            if (cars[i].goal.gameObject.activeSelf != active) cars[i].goal.gameObject.SetActive(active);
            if (cars[i].spawnPoint.gameObject.activeSelf != active) cars[i].spawnPoint.gameObject.SetActive(active);
        }
    }

    Route[] SampleRoutes(System.Random rng)
    {
        // Use the largest 4-connected Asphalt component. This guarantees that
        // every sampled start can reach its goal without crossing a different tile.
        int size = gridManager.gridSize;
        bool[,] visited = new bool[size, size];
        var largest = new List<Vector2Int>();
        var queue = new Queue<Vector2Int>();
        for (int x = 0; x < size; x++)
        for (int z = 0; z < size; z++)
        {
            if (visited[x, z] || gridManager.tileTypes[x, z] != TileType.Asphalt) continue;
            var component = new List<Vector2Int>();
            visited[x, z] = true;
            queue.Enqueue(new Vector2Int(x, z));
            while (queue.Count > 0)
            {
                Vector2Int cell = queue.Dequeue();
                component.Add(cell);
                TryEnqueueAsphalt(cell.x - 1, cell.y, size, visited, queue);
                TryEnqueueAsphalt(cell.x + 1, cell.y, size, visited, queue);
                TryEnqueueAsphalt(cell.x, cell.y - 1, size, visited, queue);
                TryEnqueueAsphalt(cell.x, cell.y + 1, size, visited, queue);
            }
            if (component.Count > largest.Count) largest = component;
        }

        // A two-cell map-edge margin keeps the car body away from the boundary.
        largest.RemoveAll(p => p.x < 2 || p.y < 2 || p.x >= size - 2 || p.y >= size - 2);
        // Validation uses the full pool so 2/5/10-car comparisons share a route
        // prefix. Training only needs endpoints for active cars, allowing every
        // training map to participate even if it cannot fit the entire pool.
        int routeCount = randomizeScenarioEachEpisode ? activeCarCount : cars.Length;
        int endpointCount = routeCount * 2;
        if (largest.Count < endpointCount)
            throw new InvalidOperationException($"Map has only {largest.Count} interior Asphalt " +
                                                $"cells in its largest component; need {endpointCount}.");

        float minSeparationCells2 = Mathf.Pow(minEndpointSeparationM / gridManager.cellSize, 2f);
        float minOwnDistanceM = smoothRouteLengthCurriculum
            ? curriculumRouteMinDistanceM : minOwnSpawnGoalDistanceM;
        float maxOwnDistanceM = CurrentRouteMaxDistanceM();
        if (maxOwnDistanceM > 0f && randomizeRoutes)
            return SampleRoutesWithDistanceBounds(rng, largest, routeCount, minSeparationCells2,
                minOwnDistanceM, maxOwnDistanceM);
        float minOwnCells2 = Mathf.Pow(minOwnDistanceM / gridManager.cellSize, 2f);
        for (int attempt = 0; attempt < placementAttempts; attempt++)
        {
            var selected = new List<Vector2Int>(endpointCount) { largest[rng.Next(largest.Count)] };
            var nearest2 = new float[largest.Count];
            for (int i = 0; i < nearest2.Length; i++) nearest2[i] = float.PositiveInfinity;
            while (selected.Count < endpointCount)
            {
                Vector2Int last = selected[selected.Count - 1];
                int bestIndex = -1;
                double bestScore = -1;
                for (int i = 0; i < largest.Count; i++)
                {
                    float d2 = (largest[i] - last).sqrMagnitude;
                    nearest2[i] = Mathf.Min(nearest2[i], d2);
                    if (nearest2[i] < minSeparationCells2) continue;
                    // Small jitter keeps placements varied while retaining wide spacing.
                    double score = nearest2[i] * (0.9 + 0.1 * rng.NextDouble());
                    if (score > bestScore) { bestScore = score; bestIndex = i; }
                }
                if (bestIndex < 0) break;
                selected.Add(largest[bestIndex]);
            }
            if (selected.Count != endpointCount) continue;

            float centerX = 0f, centerZ = 0f;
            foreach (Vector2Int p in selected) { centerX += p.x; centerZ += p.y; }
            centerX /= selected.Count;
            centerZ /= selected.Count;
            selected.Sort((a, b) => Mathf.Atan2(a.y - centerZ, a.x - centerX)
                .CompareTo(Mathf.Atan2(b.y - centerZ, b.x - centerX)));
            int offset = rng.Next(routeCount);
            var routes = new Route[routeCount];
            bool valid = true;
            for (int seat = 0; seat < routeCount; seat++)
            {
                Vector2Int start = selected[(seat + offset) % endpointCount];
                Vector2Int goal = selected[(seat + offset + routeCount) % endpointCount];
                if ((start - goal).sqrMagnitude < minOwnCells2) { valid = false; break; }
                if (rng.Next(2) == 0) { Vector2Int swap = start; start = goal; goal = swap; }
                Vector3 startWorld = CellCenter(start);
                Vector3 goalWorld = CellCenter(goal);
                routes[seat] = new Route {
                    spawn_x = startWorld.x, spawn_z = startWorld.z,
                    goal_x = goalWorld.x, goal_z = goalWorld.z,
                    heading_deg = cars[seat].TrafficRoadAlignedHeading(startWorld)
                };
            }
            if (valid) return routes;
        }
        throw new InvalidOperationException($"Could not place {routeCount} cars on connected " +
            $"Asphalt after {placementAttempts} tries (endpoint >= {minEndpointSeparationM}m, " +
            $"own start-goal >= {minOwnDistanceM}m). Lower the limits or inspect this map.");
    }

    Route[] SampleRoutesWithDistanceBounds(System.Random rng, List<Vector2Int> asphalt,
        int routeCount, float minEndpointSeparationCells2, float minDistanceM, float maxDistanceM)
    {
        // The legacy sampler deliberately pairs opposite, widely separated
        // endpoints. That is useful for hard validation, but cannot produce a
        // reliable short-route curriculum. Here each pair is sampled directly
        // within the requested Euclidean distance band, while every endpoint
        // still remains separated from every other endpoint.
        float minRouteCells2 = Mathf.Pow(minDistanceM / gridManager.cellSize, 2f);
        float maxRouteCells2 = Mathf.Pow(maxDistanceM / gridManager.cellSize, 2f);
        if (maxRouteCells2 < minRouteCells2)
            throw new InvalidOperationException("Traffic route curriculum maximum is below its minimum distance.");

        for (int attempt = 0; attempt < placementAttempts; attempt++)
        {
            var used = new List<Vector2Int>(routeCount * 2);
            var routes = new Route[routeCount];
            bool valid = true;
            for (int seat = 0; seat < routeCount && valid; seat++)
            {
                Vector2Int start = default, goal = default;
                bool pairFound = false;
                // Random candidate order is important: always taking the
                // first valid cell would create a placement bias on the grid.
                for (int choice = 0; choice < 160 && !pairFound; choice++)
                {
                    Vector2Int candidateStart = asphalt[rng.Next(asphalt.Count)];
                    if (!IsEndpointSeparated(candidateStart, used, minEndpointSeparationCells2)) continue;
                    for (int goalChoice = 0; goalChoice < 160; goalChoice++)
                    {
                        Vector2Int candidateGoal = asphalt[rng.Next(asphalt.Count)];
                        float routeCells2 = (candidateGoal - candidateStart).sqrMagnitude;
                        if (routeCells2 < minRouteCells2 || routeCells2 > maxRouteCells2) continue;
                        if (!IsEndpointSeparated(candidateGoal, used, minEndpointSeparationCells2)) continue;
                        start = candidateStart; goal = candidateGoal; pairFound = true; break;
                    }
                }
                if (!pairFound) { valid = false; break; }
                used.Add(start); used.Add(goal);
                if (rng.Next(2) == 0) { Vector2Int swap = start; start = goal; goal = swap; }
                Vector3 startWorld = CellCenter(start);
                Vector3 goalWorld = CellCenter(goal);
                routes[seat] = new Route {
                    spawn_x = startWorld.x, spawn_z = startWorld.z,
                    goal_x = goalWorld.x, goal_z = goalWorld.z,
                    heading_deg = cars[seat].TrafficRoadAlignedHeading(startWorld)
                };
            }
            if (valid) return routes;
        }
        throw new InvalidOperationException($"Could not place {routeCount} cars after {placementAttempts} attempts " +
            $"with endpoints >= {minEndpointSeparationM}m and own routes {minDistanceM}..{maxDistanceM}m.");
    }

    static bool IsEndpointSeparated(Vector2Int candidate, List<Vector2Int> existing, float minimumDistanceCells2)
    {
        foreach (Vector2Int endpoint in existing)
            if ((candidate - endpoint).sqrMagnitude < minimumDistanceCells2) return false;
        return true;
    }

    void TryEnqueueAsphalt(int x, int z, int size, bool[,] visited, Queue<Vector2Int> queue)
    {
        if (x < 0 || z < 0 || x >= size || z >= size || visited[x, z] ||
            gridManager.tileTypes[x, z] != TileType.Asphalt) return;
        visited[x, z] = true;
        queue.Enqueue(new Vector2Int(x, z));
    }

    Vector3 CellCenter(Vector2Int cell)
    {
        float sizeM = gridManager.gridSize * gridManager.cellSize;
        return new Vector3(-sizeM * 0.5f + (cell.x + 0.5f) * gridManager.cellSize,
                           0f,
                           -sizeM * 0.5f + (cell.y + 0.5f) * gridManager.cellSize);
    }

    public void OnAgentEpisodeBegin(CarAgent car)
    {
        if (!enabled) return;
        // Enabling a pooled car can synchronously invoke this callback. Its
        // route is installed by PlacePreparedRecoveryGroup immediately after
        // activation, so doing nothing here prevents any stale-route spawn.
        if (preparingRecoveryGroup) return;
        if (preparedEpisode != episodeNumber) PrepareEpisode();
        int seat = Seat(car);
        Route route = episodeRoutes[seat];
        car.PlaceForTraffic(route);
        ready[seat] = true;
        running = AllActive(ready);
        if (running) RecordTrajectoryPositions(true); // All cars at their episode spawns.
    }

    public void Finish(CarAgent car, string outcome)
    {
        int seat = Seat(car);
        if (!running || finished[seat]) return;
        finished[seat] = true;
        outcomes[seat] = outcome;
        TileType finishTile = gridManager.GetTileAt(car.transform.position);
        Academy.Instance.StatsRecorder.Add(
            $"Custom/TrafficSeat{seat}FinishTileCode", (float)finishTile,
            StatAggregationMethod.MostRecent);
        car.FreezeForTraffic(outcome);
        if (removeCarsAfterGoal) car.RemoveFromTrafficScene();
        // A traffic episode has one outcome per active car.  Keep an explicit
        // denominator and outcome counters in the Unity event logs; the
        // standard trainer console summary does not expose VehicleCollision.
        Academy.Instance.StatsRecorder.Add("Custom/CompletedVehicleEpisodes", 1f, StatAggregationMethod.Sum);
        switch (outcome)
        {
            case "Goal":
                Academy.Instance.StatsRecorder.Add("Custom/OutcomeGoal", 1f, StatAggregationMethod.Sum);
                break;
            case "VehicleCollision":
                Academy.Instance.StatsRecorder.Add("Custom/OutcomeVehicleCollision", 1f, StatAggregationMethod.Sum);
                break;
            case "Terminated":
                Academy.Instance.StatsRecorder.Add("Custom/OutcomeTerminated", 1f, StatAggregationMethod.Sum);
                break;
            case "MaxStep":
                Academy.Instance.StatsRecorder.Add("Custom/OutcomeMaxStep", 1f, StatAggregationMethod.Sum);
                break;
        }
        Academy.Instance.StatsRecorder.Add(
            $"Custom/TrafficSeat{seat}{outcome}", 1f, StatAggregationMethod.Sum);
        Academy.Instance.StatsRecorder.Add(
            $"Custom/TrafficSeat{seat}FinishStep", sharedStep, StatAggregationMethod.MostRecent);
        if (AllActive(finished)) resetPending = true;
    }

    bool AllActive(bool[] values)
    {
        for (int i = 0; i < activeCarCount; i++)
            if (!values[i]) return false;
        return true;
    }

    void FixedUpdate()
    {
        if (!running) return;
        // Defer EndEpisode until outside goal/collision callbacks.
        if (resetPending)
        {
            RecordTrajectoryPositions(true); // Capture contact/goal positions before reset.
            ResetGroup();
            return;
        }
        sharedStep++;
        RecordTrajectoryPositions(false);
        RecordCarDistances(true);
        RecordActiveTileTime();
        if (sharedStep < sharedMaxSteps) return;
        for (int i = 0; i < activeCarCount; i++)
            if (!finished[i]) cars[i].FinishTrafficMaxStep();
        RecordTrajectoryPositions(true);
        ResetGroup();
    }

    void RecordTrajectoryPositions(bool force)
    {
        if (!recordTrajectories || (!force && sharedStep % trajectoryStrideSteps != 0)) return;
        Academy.Instance.StatsRecorder.Add("Custom/TrafficTraceStep", sharedStep,
            StatAggregationMethod.MostRecent);
        for (int seat = 0; seat < activeCarCount; seat++)
        {
            Vector3 position = cars[seat].transform.position;
            Academy.Instance.StatsRecorder.Add($"Custom/TrafficSeat{seat}TraceX", position.x,
                StatAggregationMethod.MostRecent);
            Academy.Instance.StatsRecorder.Add($"Custom/TrafficSeat{seat}TraceZ", position.z,
                StatAggregationMethod.MostRecent);
        }
    }

    void RecordActiveTileTime()
    {
        for (int i = 0; i < activeCarCount; i++)
        {
            if (finished[i]) continue; // Do not count time spent parked after reaching a goal.
            int tile = (int)gridManager.GetTileAt(cars[i].transform.position);
            if (tile >= 0 && tile < TileTypeCount)
                activeTileSeconds[i, tile] += Time.fixedDeltaTime;
        }
    }

    void RecordCarDistances(bool countNearStep)
    {
        float nearestThisStep = float.PositiveInfinity;
        for (int i = 0; i < activeCarCount; i++)
        {
            if (cars[i].IsRemovedFromTrafficScene) continue;
            Vector3 a = cars[i].transform.position;
            for (int j = i + 1; j < activeCarCount; j++)
            {
                if (cars[j].IsRemovedFromTrafficScene) continue;
                Vector3 b = cars[j].transform.position;
                float dx = a.x - b.x;
                float dz = a.z - b.z;
                nearestThisStep = Mathf.Min(nearestThisStep, Mathf.Sqrt(dx * dx + dz * dz));
            }
        }
        minCarCenterDistanceM = Mathf.Min(minCarCenterDistanceM, nearestThisStep);
        if (countNearStep && nearestThisStep < nearEncounterDistanceM) nearEncounterSteps++;
    }

    [ContextMenu("Reset current traffic/recovery episode")]
    public void ForceReset()
    {
        if (!running) return;
        for (int i = 0; i < activeCarCount; i++)
            if (!finished[i]) Finish(cars[i], "Manual");
        resetPending = true;
    }

    // Exact 2D ray intersection with each other vehicle's oriented body collider.
    // This avoids scanning thousands of road tile colliders with Physics.RaycastAll.
    public bool TryVehicleHit(CarAgent viewer, Vector3 origin, Vector3 unitDirection, out float distance)
    {
        CarAgent ignored;
        return TryNearestVehicleHit(viewer, origin, unitDirection, out ignored, out distance);
    }

    // Same exact 2D body-box query as TryVehicleHit, but also exposes which moving
    // traffic car was hit.  CarAgent uses this only for optional TTC shaping; the
    // regular laser observations and collision rules remain unchanged.
    public bool TryNearestVehicleHit(CarAgent viewer, Vector3 origin, Vector3 unitDirection,
        out CarAgent hitCar, out float distance)
    {
        hitCar = null;
        distance = float.PositiveInfinity;
        for (int i = 0; i < activeCarCount; i++)
        {
            if (cars[i] == viewer || cars[i].IsRemovedFromTrafficScene) continue;
            float candidate = float.PositiveInfinity;
            TestVehicleBox(bodyColliders[i], origin, unitDirection, ref candidate);
            if (candidate < distance)
            {
                distance = candidate;
                hitCar = cars[i];
            }
        }
        if (parkedVehicle != null && parkedVehicle.gameObject.activeInHierarchy)
        {
            float candidate = float.PositiveInfinity;
            TestVehicleBox(parkedVehicleCollider, origin, unitDirection, ref candidate);
            if (candidate < distance) distance = candidate;
        }
        return !float.IsPositiveInfinity(distance);
    }

    void TestVehicleBox(BoxCollider box, Vector3 origin, Vector3 unitDirection, ref float closestDistance)
    {
        if (box == null || !box.enabled) return;
        Vector3 o = box.transform.InverseTransformPoint(origin) - box.center;
        Vector3 d = box.transform.InverseTransformVector(unitDirection);
        float inflation = vehicleSensorInflationM;
        float near = 0f, far = float.PositiveInfinity;
        if (!ClipAxis(o.x, d.x, -box.size.x * 0.5f - inflation, box.size.x * 0.5f + inflation, ref near, ref far) ||
            !ClipAxis(o.z, d.z, -box.size.z * 0.5f - inflation, box.size.z * 0.5f + inflation, ref near, ref far))
            return;
        if (near < closestDistance) closestDistance = near;
    }

    static bool ClipAxis(float origin, float direction, float min, float max, ref float near, ref float far)
    {
        if (Mathf.Abs(direction) < 1e-6f) return origin >= min && origin <= max;
        float a = (min - origin) / direction;
        float b = (max - origin) / direction;
        if (a > b) { float swap = a; a = b; b = swap; }
        near = Mathf.Max(near, a);
        far = Mathf.Min(far, b);
        return near <= far && far >= 0f;
    }

    void CreateParkedVehicle()
    {
        if (parkedVehicle != null) return;
        var go = GameObject.CreatePrimitive(PrimitiveType.Cube);
        go.name = "Recovery parked traffic vehicle";
        go.transform.SetParent(transform, false);
        go.AddComponent<TrafficVehicleObstacle>();
        var body = bodyColliders != null && bodyColliders.Length > 0 ? bodyColliders[0] : null;
        Vector3 size = body != null ? body.bounds.size : new Vector3(2f, 1f, 4f);
        go.transform.localScale = new Vector3(Mathf.Max(1f, size.x), Mathf.Max(0.75f, size.y), Mathf.Max(2f, size.z));
        var rigidbody = go.AddComponent<Rigidbody>();
        rigidbody.isKinematic = true;
        rigidbody.useGravity = false;
        parkedVehicle = go.GetComponent<TrafficVehicleObstacle>();
        parkedVehicleCollider = go.GetComponent<BoxCollider>();
        go.SetActive(false);
    }

    void EnableParkedVehicle(Vector3 position, float headingDeg)
    {
        if (parkedVehicle == null) CreateParkedVehicle();
        // Align its visual/collider with the road and use the normal car body's height.
        float y = bodyColliders != null && bodyColliders.Length > 0
            ? bodyColliders[0].bounds.center.y : 0.6f;
        parkedVehicle.transform.SetPositionAndRotation(new Vector3(position.x, y, position.z),
            Quaternion.Euler(0f, headingDeg, 0f));
        parkedVehicle.gameObject.SetActive(true);
    }

    void DisableParkedVehicle()
    {
        if (parkedVehicle != null) parkedVehicle.gameObject.SetActive(false);
    }

    void ResetGroup()
    {
        if (!running) return;
        running = false;
        RecordCarDistances(false); // Include contact positions before a collision reset.
        Debug.Log($"[Traffic] Episode {episodeNumber} ended at shared step {sharedStep}: " +
                  string.Join(", ", outcomes));
        Academy.Instance.StatsRecorder.Add("Custom/TrafficSharedSteps", sharedStep,
            StatAggregationMethod.MostRecent);
        Academy.Instance.StatsRecorder.Add("Custom/TrafficMinCenterDistanceM",
            minCarCenterDistanceM, StatAggregationMethod.MostRecent);
        Academy.Instance.StatsRecorder.Add("Custom/TrafficNearEncounterSteps",
            nearEncounterSteps, StatAggregationMethod.MostRecent);
        Academy.Instance.StatsRecorder.Add("Custom/TrafficMapIndex", gridManager.LastLoadedVoronoiMapIndex,
            StatAggregationMethod.MostRecent);
        Academy.Instance.StatsRecorder.Add("Custom/TrafficActiveCars", activeCarCount,
            StatAggregationMethod.MostRecent);
        Academy.Instance.StatsRecorder.Add("Custom/TrafficRandomPlacement", randomizeRoutes ? 1f : 0f,
            StatAggregationMethod.MostRecent);
        Academy.Instance.StatsRecorder.Add("Custom/TrafficRemoveCompletedGoalCars",
            removeCarsAfterGoal ? 1f : 0f, StatAggregationMethod.MostRecent);
        Academy.Instance.StatsRecorder.Add("Custom/TrafficRemoveFinishedCars",
            removeCarsAfterGoal ? 1f : 0f, StatAggregationMethod.MostRecent);
        Academy.Instance.StatsRecorder.Add("Custom/TrafficPlacementSeed", placementSeed,
            StatAggregationMethod.MostRecent);
        for (int seat = 0; seat < activeCarCount; seat++)
        {
            Route route = episodeRoutes[seat];
            Academy.Instance.StatsRecorder.Add($"Custom/TrafficSeat{seat}SpawnX", route.spawn_x,
                StatAggregationMethod.MostRecent);
            Academy.Instance.StatsRecorder.Add($"Custom/TrafficSeat{seat}SpawnZ", route.spawn_z,
                StatAggregationMethod.MostRecent);
            Academy.Instance.StatsRecorder.Add($"Custom/TrafficSeat{seat}GoalX", route.goal_x,
                StatAggregationMethod.MostRecent);
            Academy.Instance.StatsRecorder.Add($"Custom/TrafficSeat{seat}GoalZ", route.goal_z,
                StatAggregationMethod.MostRecent);
            Academy.Instance.StatsRecorder.Add($"Custom/TrafficSeat{seat}HeadingDeg", route.heading_deg,
                StatAggregationMethod.MostRecent);
            Academy.Instance.StatsRecorder.Add($"Custom/TrafficSeat{seat}SpawnTileCode",
                (float)gridManager.GetTileAt(new Vector3(route.spawn_x, 0f, route.spawn_z)),
                StatAggregationMethod.MostRecent);
            Academy.Instance.StatsRecorder.Add($"Custom/TrafficSeat{seat}GoalTileCode",
                (float)gridManager.GetTileAt(new Vector3(route.goal_x, 0f, route.goal_z)),
                StatAggregationMethod.MostRecent);
            for (int tile = 0; tile < TileTypeCount; tile++)
            {
                Academy.Instance.StatsRecorder.Add(
                    $"Custom/TrafficSeat{seat}TileSeconds{(TileType)tile}",
                    activeTileSeconds[seat, tile], StatAggregationMethod.MostRecent);
            }
        }
        episodeNumber++;
        preparedEpisode = -1;
        for (int i = 0; i < activeCarCount; i++)
            cars[i].EndTrafficEpisode();
    }
}
