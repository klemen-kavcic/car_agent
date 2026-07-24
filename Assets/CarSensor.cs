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

    void Awake() => BuildOffsets();

    // Keeps currentTotalPoints correct in the Inspector the moment any count/shape field changes,
    // without needing Play mode - just recomputes the offsets, same as BuildOffsets().
    void OnValidate() => BuildOffsets();

    void BuildOffsets()
    {
        sensorOffsets = sensorShape == Shape.Fan ? BuildFanOffsets() : BuildGridOffsets();
        readings = new TileType[sensorOffsets.Length];
        currentTotalPoints = sensorOffsets.Length;
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
    Vector2[] BuildFanOffsets()
    {
        var offsets = new List<Vector2>();
        float[] distances = { ring1Distance, ring2Distance, ring3Distance };
        int[] frontCounts = { ring1FrontPoints, ring2FrontPoints, ring3FrontPoints };
        int[] rearCounts = { ring1RearPoints, ring2RearPoints, ring3RearPoints };

        for (int ring = 0; ring < distances.Length; ring++)
        {
            AddArc(offsets, distances[ring], frontCounts[ring], frontHalfAngleDegrees, forward: true);
            AddArc(offsets, distances[ring], rearCounts[ring], rearHalfAngleDegrees, forward: false);
        }
        AddStraightLine(offsets, ring1Distance, ring2Distance, pointsBetweenRing1And2);
        AddStraightLine(offsets, ring2Distance, ring3Distance, pointsBetweenRing2And3);
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
