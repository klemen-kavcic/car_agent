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
    [Tooltip("Allow Python evaluation to choose a case with traffic_scenario_index.")]
    public bool useEnvironmentScenarioIndex = true;
    [Min(0f)] public float vehicleSensorInflationM = 0.1f;
    [Min(0.1f)] public float nearEncounterDistanceM = 10f;

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
        activeCarCount = defaultActiveCarCount;
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

        // A traffic manager only has useful shared-episode semantics when
        // there are at least two vehicles.  In particular, a one-seat shared
        // manager can leave the trainer with an indefinitely open episode in
        // headless multi-environment training.  Treat the command-line
        // one-car case as genuine solo pretraining instead: retain the same
        // GridManager and 89-observation car prefab, but hand episode resets
        // back to CarAgent's normal goal/terminal/MaxStep implementation.
        // Multi-agent runs (2, 5, and 10 cars) do not enter this branch.
        if (activeCarCount == 1)
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
        ready = new bool[activeCarCount];
        finished = new bool[activeCarCount];
        outcomes = new string[activeCarCount];
        bodyColliders = new BoxCollider[activeCarCount];
        activeTileSeconds = new float[activeCarCount, TileTypeCount];
        for (int i = 0; i < cars.Length; i++)
        {
            if (cars[i] == null) throw new InvalidOperationException("Traffic car is missing at seat " + i);
            cars[i].trafficManager = this;
            cars[i].trafficSeatIndex = i;
            cars[i].MaxStep = 0; // One clock for all cars; no individual auto-reset.
            if (i >= activeCarCount) continue;
            bodyColliders[i] = cars[i].GetComponent<BoxCollider>();
            if (bodyColliders[i] == null)
                throw new InvalidOperationException("Traffic car needs a body BoxCollider at seat " + i);
        }
        // Extra cars are pooled in the build but have no Agent behavior/physics until selected.
        for (int i = 0; i < cars.Length; i++)
        {
            bool active = i < activeCarCount;
            cars[i].gameObject.SetActive(active);
            cars[i].goal.gameObject.SetActive(active);
            cars[i].spawnPoint.gameObject.SetActive(active);
        }
        Debug.Log($"[Traffic] Active cars: {activeCarCount}/{cars.Length}; " +
                  $"scenario count: {collection.scenarios.Length}.");
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
        Scenario scenario = SelectedScenario();
        if (!randomizeRoutes && (scenario.routes == null || scenario.routes.Length < activeCarCount))
            throw new InvalidOperationException("Traffic scenario does not have a route for every car.");
        gridManager.forceAllNormal = false;
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
        Debug.Log($"[Traffic] Episode {episodeNumber}, scenario {scenario.scenario_id}, " +
                  $"map {scenario.map_file}, cars {activeCarCount}, placement seed {placementSeed}.");
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
        float minOwnCells2 = Mathf.Pow(minOwnSpawnGoalDistanceM / gridManager.cellSize, 2f);
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
            $"own start-goal >= {minOwnSpawnGoalDistanceM}m). Lower the limits or inspect this map.");
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
        if (preparedEpisode != episodeNumber) PrepareEpisode();
        int seat = Seat(car);
        Route route = episodeRoutes[seat];
        car.PlaceForTraffic(route);
        ready[seat] = true;
        running = Array.TrueForAll(ready, isReady => isReady);
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
        if (Array.TrueForAll(finished, isFinished => isFinished)) resetPending = true;
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
            Vector3 a = cars[i].transform.position;
            for (int j = i + 1; j < activeCarCount; j++)
            {
                Vector3 b = cars[j].transform.position;
                float dx = a.x - b.x;
                float dz = a.z - b.z;
                nearestThisStep = Mathf.Min(nearestThisStep, Mathf.Sqrt(dx * dx + dz * dz));
            }
        }
        minCarCenterDistanceM = Mathf.Min(minCarCenterDistanceM, nearestThisStep);
        if (countNearStep && nearestThisStep < nearEncounterDistanceM) nearEncounterSteps++;
    }

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
        distance = float.PositiveInfinity;
        for (int i = 0; i < activeCarCount; i++)
        {
            if (cars[i] == viewer) continue;
            BoxCollider box = bodyColliders[i];
            Vector3 o = box.transform.InverseTransformPoint(origin) - box.center;
            Vector3 d = box.transform.InverseTransformVector(unitDirection);
            float inflation = vehicleSensorInflationM;
            float near = 0f, far = float.PositiveInfinity;
            if (!ClipAxis(o.x, d.x, -box.size.x * 0.5f - inflation, box.size.x * 0.5f + inflation, ref near, ref far) ||
                !ClipAxis(o.z, d.z, -box.size.z * 0.5f - inflation, box.size.z * 0.5f + inflation, ref near, ref far))
                continue;
            if (near < distance) distance = near;
        }
        return !float.IsPositiveInfinity(distance);
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
