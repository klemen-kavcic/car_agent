using UnityEngine;
using System.Collections.Generic;

// Editor play-mode visualisation for traffic experiments.  TrailRenderer has
// no collider and is never queried by CarAgent's sensors, so this component
// is purely a viewing aid.
public sealed class TrafficCarTrails : MonoBehaviour
{
    [Tooltip("Keep the path for this many seconds. Set to a large value to keep a full episode path.")]
    [Min(0.1f)] public float trailTime = 90f;
    [Min(0.01f)] public float trailWidth = 0.13f;
    [Tooltip("A movement farther than this in one frame is treated as an episode reset and clears the old path.")]
    [Min(1f)] public float teleportClearDistance = 4f;

    // High-contrast colours that remain distinguishable over asphalt, gravel,
    // and red terminal tiles.  The seat index always maps to the same colour.
    static readonly Color[] SeatColors =
    {
        new Color(0.10f, 0.75f, 1.00f), // cyan
        new Color(1.00f, 0.42f, 0.10f), // orange
        new Color(0.30f, 0.92f, 0.35f), // green
        new Color(0.85f, 0.25f, 0.92f), // purple
        new Color(1.00f, 0.88f, 0.10f), // yellow
        new Color(1.00f, 0.25f, 0.45f), // pink-red
        new Color(0.20f, 0.55f, 1.00f), // blue
        new Color(0.60f, 1.00f, 0.35f), // lime
        new Color(1.00f, 0.58f, 0.78f), // light pink
        new Color(0.45f, 0.85f, 0.82f), // teal
    };

    TrafficScenarioManager manager;
    TrailRenderer[] trails;
    Vector3[] lastPositions;
    Material[] trailMaterials;
    readonly List<Material> frontMaterials = new List<Material>();

    void Awake()
    {
        manager = GetComponent<TrafficScenarioManager>();
        if (manager == null || manager.cars == null) return;

        trails = new TrailRenderer[manager.cars.Length];
        lastPositions = new Vector3[manager.cars.Length];
        trailMaterials = new Material[manager.cars.Length];

        for (int seat = 0; seat < manager.cars.Length; seat++)
        {
            CarAgent car = manager.cars[seat];
            if (car == null) continue;

            Color color = SeatColors[seat % SeatColors.Length];
            trailMaterials[seat] = CreateTrailMaterial(color);
            trails[seat] = CreateTrail(car.transform, color, trailMaterials[seat]);
            SetFrontColor(car.transform, color);
            lastPositions[seat] = car.transform.position;
            SetGoalColor(car.goal, color);
        }
    }

    void LateUpdate()
    {
        if (manager == null || manager.cars == null || trails == null) return;

        for (int seat = 0; seat < manager.cars.Length; seat++)
        {
            CarAgent car = manager.cars[seat];
            TrailRenderer trail = trails[seat];
            if (car == null || trail == null) continue;

            Vector3 position = car.transform.position;
            // Shared-episode resets teleport all cars to new routes. Do not
            // draw a misleading line across the map between those episodes.
            if ((position - lastPositions[seat]).sqrMagnitude > teleportClearDistance * teleportClearDistance)
                trail.Clear();
            lastPositions[seat] = position;
        }
    }

    TrailRenderer CreateTrail(Transform car, Color color, Material material)
    {
        Transform holder = car.Find("VisualTrail");
        if (holder == null)
        {
            var trailObject = new GameObject("VisualTrail");
            holder = trailObject.transform;
            holder.SetParent(car, false);
        }
        // Trail positions are recorded in world space; the small y offset
        // prevents z-fighting with the road without interacting with it.
        holder.localPosition = new Vector3(0f, 0.06f, 0f);
        holder.localRotation = Quaternion.identity;

        TrailRenderer trail = holder.GetComponent<TrailRenderer>();
        if (trail == null) trail = holder.gameObject.AddComponent<TrailRenderer>();
        trail.time = trailTime;
        trail.minVertexDistance = 0.08f;
        trail.widthMultiplier = trailWidth;
        trail.alignment = LineAlignment.View;
        trail.textureMode = LineTextureMode.Stretch;
        trail.emitting = true;
        // Use a separate material per seat.  The URP Unlit shader then gives
        // an unambiguous colour even when the renderer ignores vertex tint.
        trail.material = material;
        trail.startColor = new Color(color.r, color.g, color.b, 0.95f);
        trail.endColor = new Color(color.r, color.g, color.b, 0.18f);
        return trail;
    }

    static Material CreateTrailMaterial(Color color)
    {
        Shader shader = Shader.Find("Universal Render Pipeline/Unlit");
        if (shader == null) shader = Shader.Find("Sprites/Default");
        var material = new Material(shader) { hideFlags = HideFlags.DontSave };
        if (material.HasProperty("_BaseColor")) material.SetColor("_BaseColor", color);
        if (material.HasProperty("_Color")) material.SetColor("_Color", color);
        return material;
    }

    static void SetGoalColor(Transform goal, Color color)
    {
        if (goal == null) return;
        foreach (Renderer renderer in goal.GetComponentsInChildren<Renderer>(true))
        {
            var properties = new MaterialPropertyBlock();
            renderer.GetPropertyBlock(properties);
            foreach (Material material in renderer.sharedMaterials)
            {
                if (material == null) continue;
                if (material.HasProperty("_BaseColor")) properties.SetColor("_BaseColor", color);
                if (material.HasProperty("_Color")) properties.SetColor("_Color", color);
            }
            renderer.SetPropertyBlock(properties);
        }

        // Some imported goal shaders do not expose a tintable colour.  Add a
        // small marker above the goal as a reliable, highly visible match for
        // its car's trail.  It has no collider, therefore cannot be detected
        // by physics or laser observations.
        Transform marker = goal.Find("VisualGoalColor");
        if (marker == null)
        {
            GameObject markerObject = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            markerObject.name = "VisualGoalColor";
            marker = markerObject.transform;
            marker.SetParent(goal, false);
            Collider markerCollider = markerObject.GetComponent<Collider>();
            // Disable immediately, then remove it. This guarantees it cannot
            // participate in even the first physics/sensor update.
            markerCollider.enabled = false;
            Object.Destroy(markerCollider);
        }
        // A broad, raised marker stays readable from the top-down Game view;
        // the original goal mesh can be quite small and uses a dark material.
        marker.localPosition = new Vector3(0f, 0.18f, 0f);
        marker.localRotation = Quaternion.identity;
        marker.localScale = new Vector3(2.4f, 0.045f, 2.4f);
        Renderer markerRenderer = marker.GetComponent<Renderer>();
        Material markerMaterial = CreateTrailMaterial(color);
        markerMaterial.hideFlags = HideFlags.DontSave;
        markerRenderer.material = markerMaterial;
    }

    void SetFrontColor(Transform car, Color color)
    {
        bool recoloredImportedFront = false;
        foreach (Renderer renderer in car.GetComponentsInChildren<Renderer>(true))
        {
            // The imported Tesla hierarchy calls the relevant objects
            // "car front", "car front.001", and so on. Do not touch wheels,
            // windows, lights, or rear body panels.
            if (!renderer.name.ToLowerInvariant().Contains("car front")) continue;

            recoloredImportedFront = true;

            Material[] original = renderer.sharedMaterials;
            var coloured = new Material[original.Length];
            for (int i = 0; i < original.Length; i++)
            {
                if (original[i] == null)
                {
                    coloured[i] = null;
                    continue;
                }

                Material copy = new Material(original[i]) { hideFlags = HideFlags.DontSave };
                if (copy.HasProperty("_BaseColor")) copy.SetColor("_BaseColor", color);
                if (copy.HasProperty("_Color")) copy.SetColor("_Color", color);
                coloured[i] = copy;
                frontMaterials.Add(copy);
            }
            renderer.sharedMaterials = coloured;
        }

        // Some scene copies expose the imported mesh under a different child
        // name, or use a material whose colour cannot be tinted.  Add a small
        // unlit hood panel as a guaranteed, purely visual front indicator in
        // those cases (and also when the mesh tint succeeded, for visibility
        // from the top-down camera).  It deliberately has no collider.
        SetFrontMarker(car, color, recoloredImportedFront);
    }

    void SetFrontMarker(Transform car, Color color, bool recoloredImportedFront)
    {
        Transform marker = car.Find("VisualFrontColor");
        if (marker == null)
        {
            GameObject markerObject = GameObject.CreatePrimitive(PrimitiveType.Cube);
            markerObject.name = "VisualFrontColor";
            marker = markerObject.transform;
            marker.SetParent(car, false);
            Collider markerCollider = markerObject.GetComponent<Collider>();
            markerCollider.enabled = false;
            Object.Destroy(markerCollider);
        }

        // Derive the marker dimensions from the actual rendered body rather
        // than assuming a particular imported-car scale.  The CarAgent local
        // +Z axis is the driving/hood direction.
        bool haveBounds = false;
        Bounds localBounds = new Bounds();
        foreach (Renderer renderer in car.GetComponentsInChildren<Renderer>(true))
        {
            if (renderer.transform == marker || renderer.name == "VisualTrail") continue;
            Bounds world = renderer.bounds;
            for (int x = 0; x <= 1; x++)
            for (int y = 0; y <= 1; y++)
            for (int z = 0; z <= 1; z++)
            {
                Vector3 point = car.InverseTransformPoint(new Vector3(
                    x == 0 ? world.min.x : world.max.x,
                    y == 0 ? world.min.y : world.max.y,
                    z == 0 ? world.min.z : world.max.z));
                if (!haveBounds) { localBounds = new Bounds(point, Vector3.zero); haveBounds = true; }
                else localBounds.Encapsulate(point);
            }
        }

        if (!haveBounds)
        {
            localBounds = new Bounds(new Vector3(0f, 0.35f, 0f), new Vector3(1.4f, 0.5f, 3f));
        }

        float width = Mathf.Max(0.15f, localBounds.size.x * 0.58f);
        float length = Mathf.Max(0.15f, localBounds.size.z * (recoloredImportedFront ? 0.18f : 0.27f));
        marker.localPosition = new Vector3(
            localBounds.center.x,
            localBounds.max.y + 0.025f,
            localBounds.max.z - localBounds.size.z * 0.22f);
        marker.localRotation = Quaternion.identity;
        marker.localScale = new Vector3(width, 0.035f, length);

        Renderer markerRenderer = marker.GetComponent<Renderer>();
        Material markerMaterial = CreateTrailMaterial(color);
        markerMaterial.hideFlags = HideFlags.DontSave;
        markerRenderer.material = markerMaterial;
        frontMaterials.Add(markerMaterial);
    }

    void OnDestroy()
    {
        if (trailMaterials == null) return;
        foreach (Material material in trailMaterials)
            if (material != null) Destroy(material);
        foreach (Material material in frontMaterials)
            if (material != null) Destroy(material);
    }
}
