using UnityEngine;
using Unity.MLAgents.Sensors;

// Twelve continuous, car-relative tile lasers: seven evenly-spaced forward rays from -45 to +45
// degrees plus the five existing side/rear rays. Each direction reports the first special tile
// (Slippery, SpeedLimited or Terminal) plus its distance, and one road-boundary distance.
// Traversal is through GridManager.tileTypes, rather than Physics.Raycast, so it is exact at
// tile borders, works on every generated map, and regards the map edge as Terminal.
public class LaserTileSensor : MonoBehaviour
{
    public const int DirectionCount = 12;
    public const int ObservationsPerDirection = 5; // special type one-hot (3), special distance, road boundary distance
    public const int ObservationCount = DirectionCount * ObservationsPerDirection;

    [Tooltip("Grid whose logical tile map the lasers traverse.")]
    public GridManager gridManager;
    private CarAgent agent;

    [Header("Distance normalization")]
    [Tooltip("Characteristic distance S for special tiles in d/(d+S). YAML can override this per speed-cap condition.")]
    [Min(0.001f)] public float specialDistanceScaleM = 50f;
    [Tooltip("Characteristic distance S for the Asphalt/non-Asphalt road boundary in d/(d+S). The generated roads are about 6 m wide, so 3 m represents their half-width and maps a centred side-boundary reading to about 0.5.")]
    [Min(0.001f)] public float roadBoundaryDistanceScaleM = 3f;

    [Tooltip("Draw the exact grid-traversal rays while this object is selected during Play mode.")]
    public bool showGizmos = true;

    [Header("Runtime visualization")]
    [Tooltip("Draw laser rays in the Game view as well as Scene-view gizmos. This is a visual aid only and is ignored in batch/HPC runs.")]
    public bool showRuntimeLasers;
    [Min(0.001f)] public float runtimeLineWidth = 0.07f;
    [Min(0f)] public float runtimeLineHeight = 0.16f;

    private bool runtimeLasersForcedVisible;
    private GameObject runtimeLaserRoot;
    private LineRenderer[] runtimeLines;
    private Material runtimeLineMaterial;

    // Clockwise, car-local directions. Forward coverage is dense at 15-degree intervals; the
    // original right/back/left directions remain. These rotate automatically with the car.
    private static readonly Vector3[] LocalDirections =
    {
        DirectionFromDegrees(-45f),
        DirectionFromDegrees(-30f),
        DirectionFromDegrees(-15f),
        DirectionFromDegrees(0f),
        DirectionFromDegrees(15f),
        DirectionFromDegrees(30f),
        DirectionFromDegrees(45f),
        new Vector3(1f, 0f, 0f),
        new Vector3(1f, 0f, -1f).normalized,
        new Vector3(0f, 0f, -1f),
        new Vector3(-1f, 0f, -1f).normalized,
        new Vector3(-1f, 0f, 0f),
    };

    private static Vector3 DirectionFromDegrees(float degrees)
    {
        float radians = degrees * Mathf.Deg2Rad;
        return new Vector3(Mathf.Sin(radians), 0f, Mathf.Cos(radians));
    }

    // Kept as a method rather than exposing the backing array so the overlay can share the exact
    // observation ordering without being able to accidentally modify it.
    public static Vector3 GetLocalDirection(int directionIndex)
    {
        return LocalDirections[directionIndex];
    }

    // ManualDrivingView uses this so the third-person test view gets world-space rays without
    // requiring the Inspector debug flag to be enabled for every training build.
    public void SetRuntimeLasersForcedVisible(bool visible)
    {
        runtimeLasersForcedVisible = visible;
        UpdateRuntimeLasers();
    }

    void LateUpdate()
    {
        UpdateRuntimeLasers();
    }

    void UpdateRuntimeLasers()
    {
        bool shouldShow = Application.isPlaying && !Application.isBatchMode &&
            (showRuntimeLasers || runtimeLasersForcedVisible) && gridManager != null && gridManager.tileTypes != null;
        if (!shouldShow)
        {
            if (runtimeLaserRoot != null) runtimeLaserRoot.SetActive(false);
            return;
        }

        EnsureRuntimeLines();
        runtimeLaserRoot.SetActive(true);
        for (int i = 0; i < DirectionCount; i++)
        {
            Vector3 direction = transform.TransformDirection(LocalDirections[i]);
            Trace(transform.position, direction, out TileType type, out float specialDistance, out _);
            Color color = type == TileType.Terminal ? Color.red :
                type == TileType.Slippery ? Color.cyan : Color.green;
            Vector3 start = transform.position + Vector3.up * runtimeLineHeight;
            Vector3 end = start + direction.normalized * specialDistance;
            LineRenderer line = runtimeLines[i];
            line.startColor = color;
            line.endColor = color;
            line.widthMultiplier = runtimeLineWidth;
            line.SetPosition(0, start);
            line.SetPosition(1, end);
        }
    }

    void EnsureRuntimeLines()
    {
        if (runtimeLaserRoot != null) return;

        runtimeLaserRoot = new GameObject("RuntimeLaserRays");
        runtimeLaserRoot.transform.SetParent(transform, false);
        runtimeLines = new LineRenderer[DirectionCount];
        Shader shader = Shader.Find("Universal Render Pipeline/Unlit");
        if (shader == null) shader = Shader.Find("Sprites/Default");
        runtimeLineMaterial = new Material(shader);

        for (int i = 0; i < DirectionCount; i++)
        {
            GameObject ray = new GameObject($"Laser{i}");
            ray.transform.SetParent(runtimeLaserRoot.transform, false);
            LineRenderer line = ray.AddComponent<LineRenderer>();
            line.material = runtimeLineMaterial;
            line.positionCount = 2;
            line.useWorldSpace = true;
            line.alignment = LineAlignment.View;
            line.numCapVertices = 3;
            runtimeLines[i] = line;
        }
    }

    void OnDestroy()
    {
        if (runtimeLineMaterial == null) return;
        if (Application.isPlaying) Destroy(runtimeLineMaterial);
        else DestroyImmediate(runtimeLineMaterial);
    }

    public void AddObservations(VectorSensor sensor)
    {
        if (gridManager == null || gridManager.tileTypes == null)
        {
            AddZeros(sensor);
            return;
        }

        Vector3 origin = transform.position;
        foreach (Vector3 localDirection in LocalDirections)
        {
            Trace(origin, transform.TransformDirection(localDirection), out TileType specialType,
                out float specialDistance, out float roadBoundaryDistance);

            // Special tile identity. Asphalt/Gravel are intentionally absent: the separate road
            // boundary reading below conveys whether to remain centred on / return to Asphalt.
            sensor.AddObservation(specialType == TileType.Slippery ? 1f : 0f);
            sensor.AddObservation(specialType == TileType.SpeedLimited ? 1f : 0f);
            sensor.AddObservation(specialType == TileType.Terminal ? 1f : 0f);
            sensor.AddObservation(NormalizeDistance(specialDistance, specialDistanceScaleM));
            sensor.AddObservation(NormalizeDistance(roadBoundaryDistance, roadBoundaryDistanceScaleM));
        }
    }

    private float NormalizeDistance(float distanceM, float scaleM)
    {
        // GridDiagonal is also Trace's explicit "no road before map edge" sentinel. Preserve 1
        // for that case; all real finite distances use the smooth, no-hard-range d/(d+S) map.
        if (distanceM >= gridManager.GridDiagonal - 0.001f) return 1f;
        float safeDistance = Mathf.Max(0f, distanceM);
        return safeDistance / (safeDistance + Mathf.Max(0.001f, scaleM));
    }

    public void AddZeros(VectorSensor sensor)
    {
        for (int i = 0; i < ObservationCount; i++) sensor.AddObservation(0f);
    }

    // The road-boundary distance has deliberately asymmetric semantics determined by the current
    // tile (which CarAgent already supplies as a 5-value one-hot):
    //   Asphalt -> distance to first non-Asphalt border along this ray (stay centred on the road)
    //   other   -> distance to first Asphalt border along this ray (recover the road direction)
    // If no Asphalt is reached before the edge while off-road, GridDiagonal encodes "no road in
    // this direction". The agent can distinguish those two cases from its current-tile one-hot.
    public void Trace(Vector3 origin, Vector3 worldDirection, out TileType specialType,
        out float specialDistance, out float roadBoundaryDistance)
    {
        specialType = TileType.Terminal;
        specialDistance = 0f;
        roadBoundaryDistance = gridManager != null ? gridManager.GridDiagonal : 1f;
        if (gridManager == null || gridManager.tileTypes == null) return;

        // Tile rays live on the XZ map plane. Ignore transient chassis pitch/roll so a bump
        // cannot shorten the horizontal distance represented by a laser.
        Vector3 direction = worldDirection;
        direction.y = 0f;
        if (direction.sqrMagnitude < 0.000001f) return;
        direction.Normalize();
        Vector2Int cell = gridManager.WorldToGrid(origin);
        if (!gridManager.IsInBounds(cell)) return; // map edge is an immediate Terminal.

        // Traffic cars are dynamic geometry, not tileTypes. Preserve the existing five
        // values per ray so a 77-observation laser checkpoint can be evaluated unchanged.
        if (agent == null) agent = GetComponent<CarAgent>();
        float vehicleDistance = float.PositiveInfinity;
        if (agent != null && agent.trafficManager != null)
            agent.trafficManager.TryVehicleHit(agent, origin, direction, out vehicleDistance);

        TileType startType = gridManager.GetTileAtCell(cell);
        bool startedOnAsphalt = startType == TileType.Asphalt;
        bool specialFound = IsSpecial(startType);
        if (specialFound)
        {
            specialType = startType;
            specialDistance = 0f;
        }
        bool roadBoundaryFound = false;

        float cellSize = gridManager.cellSize;
        Vector3 gridOrigin = gridManager.GridOrigin;
        int stepX = direction.x > 0f ? 1 : direction.x < 0f ? -1 : 0;
        int stepZ = direction.z > 0f ? 1 : direction.z < 0f ? -1 : 0;
        float deltaX = stepX == 0 ? float.PositiveInfinity : cellSize / Mathf.Abs(direction.x);
        float deltaZ = stepZ == 0 ? float.PositiveInfinity : cellSize / Mathf.Abs(direction.z);
        float boundaryX = gridOrigin.x + (stepX > 0 ? (cell.x + 1) * cellSize : cell.x * cellSize);
        float boundaryZ = gridOrigin.z + (stepZ > 0 ? (cell.y + 1) * cellSize : cell.y * cellSize);
        float nextX = stepX == 0 ? float.PositiveInfinity : (boundaryX - origin.x) / direction.x;
        float nextZ = stepZ == 0 ? float.PositiveInfinity : (boundaryZ - origin.z) / direction.z;

        // A ray crosses no more than 2*gridSize cells. The guard protects against malformed grid
        // settings while still allowing every possible route to the map edge (no sensor range cap).
        for (int crossed = 0; crossed <= gridManager.gridSize * 2 + 2; crossed++)
        {
            float distance = Mathf.Min(nextX, nextZ);
            if (vehicleDistance <= distance)
            {
                if (!specialFound)
                {
                    specialType = TileType.Terminal;
                    specialDistance = vehicleDistance;
                }
                // The separate road-boundary channel cannot see through a car. If no
                // boundary was already observed, leave the existing no-hit sentinel.
                return;
            }
            bool crossX = Mathf.Abs(nextX - distance) < 0.0001f;
            bool crossZ = Mathf.Abs(nextZ - distance) < 0.0001f;
            if (crossX) { cell.x += stepX; nextX += deltaX; }
            if (crossZ) { cell.y += stepZ; nextZ += deltaZ; }

            if (!gridManager.IsInBounds(cell))
            {
                if (!specialFound)
                {
                    specialType = TileType.Terminal;
                    specialDistance = distance;
                }
                if (startedOnAsphalt && !roadBoundaryFound)
                    roadBoundaryDistance = distance;
                return;
            }

            TileType type = gridManager.GetTileAtCell(cell);
            if (!specialFound && IsSpecial(type))
            {
                specialType = type;
                specialDistance = distance;
                specialFound = true;
            }

            if (!roadBoundaryFound && (startedOnAsphalt ? type != TileType.Asphalt : type == TileType.Asphalt))
            {
                roadBoundaryDistance = distance;
                roadBoundaryFound = true;
            }

            if (specialFound && roadBoundaryFound) return;
        }
    }

    private static bool IsSpecial(TileType type)
    {
        return type == TileType.Slippery || type == TileType.SpeedLimited || type == TileType.Terminal;
    }

    void OnDrawGizmosSelected()
    {
        // The laser component stays on the car so modes can be selected in the Inspector, but
        // only draw it when it is the observation mode actually selected for this build/scene.
        if (agent == null) agent = GetComponent<CarAgent>();
        if (agent != null && agent.tileObservationMode != CarAgent.TileObservationMode.ContinuousLasers)
            return;

        if (!showGizmos || !Application.isPlaying || gridManager == null || gridManager.tileTypes == null) return;
        foreach (Vector3 localDirection in LocalDirections)
        {
            Trace(transform.position, transform.TransformDirection(localDirection), out TileType type,
                out float specialDistance, out float roadBoundaryDistance);
            Gizmos.color = type == TileType.Terminal ? Color.red :
                type == TileType.Slippery ? Color.cyan : Color.green;
            Gizmos.DrawLine(transform.position + Vector3.up * 0.1f,
                transform.position + transform.TransformDirection(localDirection) * specialDistance + Vector3.up * 0.1f);

            // Yellow markers are road boundaries: exit points while on Asphalt, or the first
            // Asphalt entry point while on Gravel/SpeedLimited/Slippery. No marker means the
            // ray never reaches Asphalt before the map edge (encoded as GridDiagonal).
            if (roadBoundaryDistance < gridManager.GridDiagonal - 0.001f)
            {
                Gizmos.color = Color.yellow;
                Vector3 boundary = transform.position + transform.TransformDirection(localDirection) * roadBoundaryDistance;
                Gizmos.DrawWireSphere(boundary + Vector3.up * 0.12f, 0.35f);
            }
        }
    }
}
