using UnityEngine;
using System.Collections.Generic;

public class CarSensor : MonoBehaviour
{
    public enum Shape
    {
        Stadium, // full width for center 3 rows, tapers front/back  (your pattern)
        Circle,  // Euclidean circle
        Fan,     // front-emphasized concentric rings + a small rear fan for reverse
    }

    public GridManager gridManager;

    [Tooltip("Radius in sensor steps. 3 = up to 7 wide at center. Changing requires updating Space Size. Used by the Stadium/Circle shapes.")]
    public int sensorRadius = 3;

    [Tooltip("Metres between sensor points in local space. Used by the Stadium/Circle shapes.")]
    public float sensorSpacing = 3f;

    public Shape sensorShape = Shape.Stadium;

    [Header("Fan shape settings (used when Sensor Shape = Fan)")]
    [Tooltip("Distance from the car (metres) for each of the 3 concentric sensor rings. Ring 3 is typically set much farther out than rings 1-2.")]
    public float ring1Distance = 3f;
    public float ring2Distance = 8f;
    public float ring3Distance = 20f;

    [Tooltip("Number of points across the forward fan at each ring, innermost to outermost. Counts typically increase with distance, for wider far-field coverage.")]
    public int ring1FrontPoints = 3;
    public int ring2FrontPoints = 5;
    public int ring3FrontPoints = 7;

    [Tooltip("Half-angle (degrees) of the forward fan at every ring — e.g. 70 gives ~140° total forward coverage.")]
    public float frontHalfAngleDegrees = 70f;

    [Tooltip("Number of points across the rear fan at each ring, innermost to outermost. Kept small by default — reversing only needs short-range awareness, not the same far-field breadth as driving forward — but each ring is independently configurable.")]
    public int ring1RearPoints = 3;
    public int ring2RearPoints = 3;
    public int ring3RearPoints = 3;

    [Tooltip("Half-angle (degrees) of the rear fan at every ring.")]
    public float rearHalfAngleDegrees = 30f;

    [Header("Straight-ahead fill points (Fan shape only)")]
    [Tooltip("Extra points placed directly ahead (0 degrees), at evenly-spaced distances strictly between ring 1 and ring 2's distances. Fills the coverage gap along the most important (straight-ahead) direction without the cost of a full extra arc. 0 = none. Changing requires updating Space Size.")]
    public int pointsBetweenRing1And2 = 0;

    [Tooltip("Extra points placed directly ahead (0 degrees), at evenly-spaced distances strictly between ring 2 and ring 3's distances. Changing requires updating Space Size.")]
    public int pointsBetweenRing2And3 = 0;

    [Header("Info (read-only, auto-computed)")]
    [Tooltip("Current total sensor point count - set this agent's Vector Observation Space Size to match whenever you change any count above. Updates live in the Editor as you edit the fields (OnValidate), no need to enter Play mode.")]
    public int currentTotalPoints;

    // Local-space (metre) XZ offsets, computed once in BuildOffsets() from whichever shape is selected.
    private Vector2[] sensorOffsets;
    public TileType[] readings;
    public int SensorCount => sensorOffsets != null ? sensorOffsets.Length : 0;
    // Read-only view for SensorWeightOverlay.cs's live layout (same array OnDrawGizmosSelected
    // already draws from) - exposed rather than duplicated so a shape/Inspector change here is
    // automatically reflected everywhere without any second place to keep in sync.
    public System.Collections.ObjectModel.ReadOnlyCollection<Vector2> SensorOffsets =>
        sensorOffsets != null ? System.Array.AsReadOnly(sensorOffsets) : null;

    // The dead-ahead LINE through the fan, Fan shape only (null for Stadium/Circle, which have no
    // ring/arc concept) - used by carAgent.cs's dense terminal-avoidance shaping (see
    // UpdateRewardWeights/CollectObservations there). This is NOT "every front-facing point" (that
    // would be 15 points spread across 3 arcs at the current Inspector settings) - it's just the
    // single on-axis (angle=0) point from each front arc, plus the straight-ahead fill points
    // between rings, which together form one straight forward-facing line at increasing distances.
    // At the current Inspector config (front points 3/5/7, fill points 1/3) that's exactly 7 points
    // at 3/6/9/12/15/18/21m. Built in lockstep with sensorOffsets in BuildFanOffsets() so it stays
    // correct automatically if ring/point-count Inspector values change - never hardcode index
    // ranges against these counts elsewhere. An arc only contributes its on-axis point when its
    // count is odd (an even-count arc has no point exactly at angle=0); fill-line points are always
    // on-axis by construction so they're unconditionally included.
    public bool[] isForwardLineSensor;
    public int ForwardLineSensorCount { get; private set; }

    // The dead-ahead-line indices (a subset of isForwardLineSensor==true), ordered NEAREST-FIRST
    // by actual world distance (sensorOffsets[i].magnitude) - NOT the same as ascending array
    // index order, since BuildFanOffsets() appends indices ring-by-ring (front+rear arcs) and only
    // appends the straight-ahead fill points at the very end, so a fill point sitting BETWEEN
    // ring1 and ring2 has a smaller array index than ring2's own on-axis point despite being
    // farther away in some configs. Rebuilt alongside isForwardLineSensor in BuildOffsets() -
    // used by GetNearestForwardLineTerminalSeverity() below, never assume index order = distance
    // order anywhere else either.
    private int[] forwardLineIndicesByDistance;

    void Awake() => BuildOffsets();

    // Keeps currentTotalPoints correct in the Inspector the moment any count/shape field changes,
    // without needing Play mode - just recomputes the offsets, same as BuildOffsets().
    void OnValidate() => BuildOffsets();

    void BuildOffsets()
    {
        if (sensorShape == Shape.Fan)
        {
            sensorOffsets = BuildFanOffsets(out isForwardLineSensor);
        }
        else
        {
            sensorOffsets = BuildGridOffsets();
            isForwardLineSensor = null;
        }
        readings = new TileType[sensorOffsets.Length];
        currentTotalPoints = sensorOffsets.Length;
        ForwardLineSensorCount = 0;
        if (isForwardLineSensor != null)
            foreach (var f in isForwardLineSensor)
                if (f) ForwardLineSensorCount++;

        if (isForwardLineSensor != null && ForwardLineSensorCount > 0)
        {
            var indices = new List<int>(ForwardLineSensorCount);
            for (int i = 0; i < isForwardLineSensor.Length; i++)
                if (isForwardLineSensor[i]) indices.Add(i);
            indices.Sort((a, b) => sensorOffsets[a].magnitude.CompareTo(sensorOffsets[b].magnitude));
            forwardLineIndicesByDistance = indices.ToArray();
        }
        else
        {
            forwardLineIndicesByDistance = null;
        }
    }

    // Severity based on PROXIMITY of the nearest dead-ahead-line sensor currently reading
    // Terminal, not a raw count - a hazard read by the 3m point is far more urgent than one only
    // visible at 21m, and the old count-based Phi treated them identically (2 hazards read = same
    // severity whether they're the two nearest points or the two farthest). Base linear fraction is
    // (N - rank) / N (N = ForwardLineSensorCount, 7 at the current Inspector config):
    //   0                              - no dead-ahead-line sensor currently reads Terminal
    //   1/N                            - only reached by the FARTHEST dead-ahead sensor
    //   ...
    //   N/N (= 1.0)                    - the NEAREST dead-ahead sensor itself reads Terminal
    // where rank is the nearest red sensor's 0-based position in forwardLineIndicesByDistance
    // (rank 0 = nearest = severity 1.0). 0 if this isn't a Fan sensor or readings aren't populated
    // yet.
    //
    // exponent (default 1 = the plain linear fraction above, unchanged from before this parameter
    // existed) raises that fraction to a power: Mathf.Pow((N-rank)/N, exponent). exponent > 1 skews
    // weight toward the near end - e.g. at exponent=2 the farthest point's severity drops from 1/7
    // (~0.143) to (1/7)^2 (~0.020), while the nearest point stays exactly 1.0 regardless of
    // exponent (1^exponent = 1 always) - motivated by real driving urgency being nonlinear in
    // distance (a hazard 3m out deserves outsized concern vs one 18m out), at the cost of a weaker
    // early-warning gradient from the farther points the higher exponent goes. CarSensor has no
    // ML-Agents dependency by design (see class-level convention), so this is a plain parameter
    // supplied by the caller (carAgent.cs reads sensor_severity_exponent from environment_parameters
    // and passes it in) rather than read from Academy directly here.
    // 0-based position (in forwardLineIndicesByDistance, i.e. by actual world distance - rank 0 =
    // nearest) of the nearest dead-ahead-line sensor currently reading Terminal, or -1 if none are.
    // Exists as its own method (not just inlined into GetNearestForwardLineTerminalSeverity below)
    // so a caller can gate logic on RANK directly - e.g. carAgent.cs's MaxStep-outcome handling
    // treats "nearest or 2nd-nearest sensor red" as a plausibly-hazard-caused timeout, which needs
    // to stay correct regardless of sensor_severity_exponent - a fixed threshold on the SEVERITY
    // VALUE would silently mean something different at exponent=1 vs exponent=2 (rank 1's severity
    // is 6/7~=0.857 at exponent=1 but (6/7)^2~=0.735 at exponent=2), whereas "rank <= 1" means the
    // same thing - nearest or second-nearest - no matter what exponent is in effect.
    public int GetNearestForwardLineTerminalRank()
    {
        if (readings == null || forwardLineIndicesByDistance == null || forwardLineIndicesByDistance.Length == 0)
            return -1;
        int n = forwardLineIndicesByDistance.Length;
        for (int rank = 0; rank < n; rank++)
        {
            if (readings[forwardLineIndicesByDistance[rank]] == TileType.Terminal)
                return rank;
        }
        return -1;
    }

    public float GetNearestForwardLineTerminalSeverity(float exponent = 1f)
    {
        int rank = GetNearestForwardLineTerminalRank();
        if (rank < 0 || forwardLineIndicesByDistance == null)
            return 0f;
        int n = forwardLineIndicesByDistance.Length;
        float linear = (n - rank) / (float)n;
        return exponent == 1f ? linear : Mathf.Pow(linear, exponent);
    }

    Vector2[] BuildGridOffsets()
    {
        var offsets = new List<Vector2>();
        for (int dz = -sensorRadius; dz <= sensorRadius; dz++)
        {
            for (int dx = -sensorRadius; dx <= sensorRadius; dx++)
            {
                bool include = sensorShape == Shape.Circle
                    ? dx * dx + dz * dz <= sensorRadius * sensorRadius
                    : Mathf.Abs(dx) + Mathf.Max(0, Mathf.Abs(dz) - 1) <= sensorRadius;

                if (include) offsets.Add(new Vector2(dx * sensorSpacing, dz * sensorSpacing));
            }
        }
        return offsets.ToArray();
    }

    // Three concentric rings, each contributing a forward-facing fan (point count grows with
    // distance for wider far-field coverage) plus a small fixed-size rear fan for reverse —
    // reversing only needs short-range awareness, so the rear stays cheap at every ring.
    // Default point count: 3+5+7 = 15 front + 3+3+3 = 9 rear = 24 total, plus whatever
    // pointsBetweenRing1And2/pointsBetweenRing2And3 add (0 by default).
    Vector2[] BuildFanOffsets(out bool[] isForwardLine)
    {
        var offsets = new List<Vector2>();
        var forwardLine = new List<bool>();
        float[] distances = { ring1Distance, ring2Distance, ring3Distance };
        int[] frontCounts = { ring1FrontPoints, ring2FrontPoints, ring3FrontPoints };
        int[] rearCounts = { ring1RearPoints, ring2RearPoints, ring3RearPoints };

        for (int ring = 0; ring < distances.Length; ring++)
        {
            int frontStart = offsets.Count;
            AddArc(offsets, distances[ring], frontCounts[ring], frontHalfAngleDegrees, forward: true);
            // Only an ODD-count arc has a point exactly on-axis (angle=0) - that's the arc's
            // middle index. An even count straddles the centerline with no point exactly on it.
            int frontCenter = frontCounts[ring] % 2 == 1 ? frontStart + frontCounts[ring] / 2 : -1;
            for (int i = frontStart; i < offsets.Count; i++) forwardLine.Add(i == frontCenter);

            int rearStart = offsets.Count;
            AddArc(offsets, distances[ring], rearCounts[ring], rearHalfAngleDegrees, forward: false);
            for (int i = rearStart; i < offsets.Count; i++) forwardLine.Add(false); // rear never counts as forward-line
        }
        int fillStart = offsets.Count;
        AddStraightLine(offsets, ring1Distance, ring2Distance, pointsBetweenRing1And2);
        AddStraightLine(offsets, ring2Distance, ring3Distance, pointsBetweenRing2And3);
        for (int i = fillStart; i < offsets.Count; i++) forwardLine.Add(true); // always on-axis by construction

        isForwardLine = forwardLine.ToArray();
        return offsets.ToArray();
    }

    // Places `count` points directly ahead (0 degrees), at distances evenly spaced strictly
    // between distanceA and distanceB (endpoints excluded, since the rings themselves already
    // cover those exact distances via their front arc).
    void AddStraightLine(List<Vector2> offsets, float distanceA, float distanceB, int count)
    {
        if (count <= 0) return;
        for (int i = 0; i < count; i++)
        {
            float t = (i + 1) / (float)(count + 1); // strictly between 0 and 1
            float distance = Mathf.Lerp(distanceA, distanceB, t);
            offsets.Add(new Vector2(0f, distance));
        }
    }

    // Places `count` points evenly across [-halfAngle, +halfAngle] around the forward (or
    // backward) local axis, at the given distance. A single point sits exactly on-axis.
    void AddArc(List<Vector2> offsets, float distance, int count, float halfAngleDegrees, bool forward)
    {
        if (count <= 0) return;
        float baseAngle = forward ? 0f : 180f;
        for (int i = 0; i < count; i++)
        {
            float t = count == 1 ? 0f : (i / (float)(count - 1)) * 2f - 1f; // -1..1
            float rad = (baseAngle + t * halfAngleDegrees) * Mathf.Deg2Rad;
            offsets.Add(new Vector2(Mathf.Sin(rad) * distance, Mathf.Cos(rad) * distance));
        }
    }

    // Called by CarAgent.CollectObservations so readings are always current for that step
    public void UpdateReadings()
    {
        if (gridManager == null || sensorOffsets == null) return;
        for (int i = 0; i < sensorOffsets.Length; i++)
        {
            Vector3 localPos = new Vector3(sensorOffsets[i].x, 0f, sensorOffsets[i].y);
            readings[i] = gridManager.GetTileAt(transform.TransformPoint(localPos));
        }
    }

    void OnDrawGizmosSelected()
    {
        if (sensorOffsets == null) BuildOffsets();
        foreach (var offset in sensorOffsets)
        {
            Vector3 localPos = new Vector3(offset.x, 0f, offset.y);
            Vector3 worldPos = transform.TransformPoint(localPos);
            bool isOrigin = Mathf.Approximately(offset.x, 0f) && Mathf.Approximately(offset.y, 0f);
            Gizmos.color = isOrigin ? Color.yellow : Color.cyan;
            Gizmos.DrawWireSphere(worldPos, 0.5f);
        }
    }
}
