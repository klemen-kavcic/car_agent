using System;
using UnityEngine;
using Unity.MLAgents;

// Owns a shared map and a shared clock. CarAgent retains its original single-car
// behavior when no TrafficScenarioManager is assigned.
public class TrafficScenarioManager : MonoBehaviour
{
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
    public CarAgent[] cars;
    public TextAsset scenarioJson;
    [Min(0)] public int scenarioIndex = 0;
    public bool cycleScenarios = false;
    [Min(1)] public int sharedMaxSteps = 5000;
    [Tooltip("Allow Python evaluation to choose a case with traffic_scenario_index.")]
    public bool useEnvironmentScenarioIndex = true;
    [Min(0f)] public float vehicleSensorInflationM = 0.1f;

    private ScenarioCollection collection;
    private bool[] ready;
    private bool[] finished;
    private string[] outcomes;
    private BoxCollider[] bodyColliders;
    private int preparedEpisode = -1;
    private int episodeNumber;
    private int sharedStep;
    private bool running;
    private bool resetPending;

    public int SharedStep => sharedStep;
    public int SharedMaxSteps => sharedMaxSteps;
    public bool IsFinished(CarAgent car) => finished != null && finished[Seat(car)];

    void Awake()
    {
        if (scenarioJson == null || gridManager == null || cars == null || cars.Length < 2)
        {
            Debug.LogError("[Traffic] Assign a grid, at least two cars and a scenario JSON.", this);
            enabled = false;
            return;
        }
        collection = JsonUtility.FromJson<ScenarioCollection>(scenarioJson.text);
        if (collection?.scenarios == null || collection.scenarios.Length == 0)
        {
            Debug.LogError("[Traffic] Scenario JSON has no scenarios.", this);
            enabled = false;
            return;
        }
        ready = new bool[cars.Length];
        finished = new bool[cars.Length];
        outcomes = new string[cars.Length];
        bodyColliders = new BoxCollider[cars.Length];
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
    }

    int Seat(CarAgent car)
    {
        int seat = car.trafficSeatIndex;
        if (seat < 0 || seat >= cars.Length || cars[seat] != car)
            throw new InvalidOperationException("Car is not registered with this traffic manager.");
        return seat;
    }

    Scenario SelectedScenario()
    {
        int index = cycleScenarios ? (scenarioIndex + episodeNumber) % collection.scenarios.Length : scenarioIndex;
        if (useEnvironmentScenarioIndex)
        {
            float overrideIndex = Academy.Instance.EnvironmentParameters.GetWithDefault("traffic_scenario_index", -1f);
            if (overrideIndex >= 0f) index = Mathf.RoundToInt(overrideIndex);
        }
        return collection.scenarios[Mathf.Clamp(index, 0, collection.scenarios.Length - 1)];
    }

    void PrepareEpisode()
    {
        Scenario scenario = SelectedScenario();
        if (scenario.routes == null || scenario.routes.Length < cars.Length)
            throw new InvalidOperationException("Traffic scenario does not have a route for every car.");
        gridManager.forceAllNormal = false;
        gridManager.useValidationMaps = true;
        gridManager.forcedMapIndex = scenario.map_index;
        gridManager.Regenerate();
        Array.Clear(ready, 0, ready.Length);
        Array.Clear(finished, 0, finished.Length);
        Array.Clear(outcomes, 0, outcomes.Length);
        sharedStep = 0;
        running = false;
        resetPending = false;
        preparedEpisode = episodeNumber;
        Debug.Log($"[Traffic] Episode {episodeNumber}, scenario {scenario.scenario_id}, map {scenario.map_file}.");
    }

    public void OnAgentEpisodeBegin(CarAgent car)
    {
        if (!enabled) return;
        if (preparedEpisode != episodeNumber) PrepareEpisode();
        int seat = Seat(car);
        Route route = SelectedScenario().routes[seat];
        car.PlaceForTraffic(route);
        ready[seat] = true;
        running = Array.TrueForAll(ready, isReady => isReady);
    }

    public void Finish(CarAgent car, string outcome)
    {
        int seat = Seat(car);
        if (!running || finished[seat]) return;
        finished[seat] = true;
        outcomes[seat] = outcome;
        car.FreezeForTraffic(outcome);
        Academy.Instance.StatsRecorder.Add(
            $"Custom/TrafficSeat{seat}{outcome}", 1f, StatAggregationMethod.Sum);
        if (Array.TrueForAll(finished, isFinished => isFinished)) resetPending = true;
    }

    void FixedUpdate()
    {
        if (!running) return;
        // Defer EndEpisode until outside goal/collision callbacks.
        if (resetPending)
        {
            ResetGroup();
            return;
        }
        sharedStep++;
        if (sharedStep < sharedMaxSteps) return;
        for (int i = 0; i < cars.Length; i++)
            if (!finished[i]) cars[i].FinishTrafficMaxStep();
        ResetGroup();
    }

    public void ForceReset()
    {
        if (!running) return;
        for (int i = 0; i < cars.Length; i++)
            if (!finished[i]) Finish(cars[i], "Manual");
        resetPending = true;
    }

    // Exact 2D ray intersection with each other vehicle's oriented body collider.
    // This avoids scanning thousands of road tile colliders with Physics.RaycastAll.
    public bool TryVehicleHit(CarAgent viewer, Vector3 origin, Vector3 unitDirection, out float distance)
    {
        distance = float.PositiveInfinity;
        for (int i = 0; i < cars.Length; i++)
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
        Debug.Log($"[Traffic] Episode {episodeNumber} ended at shared step {sharedStep}: " +
                  string.Join(", ", outcomes));
        Academy.Instance.StatsRecorder.Add("Custom/TrafficSharedSteps", sharedStep,
            StatAggregationMethod.MostRecent);
        episodeNumber++;
        preparedEpisode = -1;
        for (int i = 0; i < cars.Length; i++)
            cars[i].EndTrafficEpisode();
    }
}
