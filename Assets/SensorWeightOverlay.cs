using System;
using System.IO;
using UnityEngine;
using UnityEngine.UI;

// Live "which sensor is the network reacting to" HUD: a fixed corner panel (car-relative, not
// world-position - forward always points up) with one dot per CarSensor fan point, colored by the
// tile type it's currently reading and sized by how much that reading pulls on the first hidden
// layer, given the CURRENTLY sensed tile type at that point (see export_sensor_weights.py's header
// comment for the exact metric and why a plain "active one-hot column's weight norm" isn't quite
// right once the observation normalizer is accounted for).
//
// Play-mode only, and only meaningful with the Agent's Behavior Parameters set to Inference Only
// against a real checkpoint - CarSensor.readings is populated by CollectObservations, which only
// runs during an active decision loop. During actual HPC training Unity never runs the network at
// all (the external Python trainer does), so this has nothing useful to show then.
//
// Workflow: run `python export_sensor_weights.py --checkpoint <run_dir>/CarAgent-<step>.onnx` for
// whichever checkpoint is assigned to this Agent's Behavior Parameters, then point weightsJsonPath
// at the resulting .sensor_weights.json before entering Play mode. A mismatched sensor_count
// (checkpoint's fan shape != this scene's current CarSensor Inspector settings) is caught and
// logged rather than silently misreading the table.
public class SensorWeightOverlay : MonoBehaviour
{
    [Tooltip("Path to the *.sensor_weights.json written by export_sensor_weights.py for the " +
             "checkpoint currently assigned to this Agent's Behavior Parameters.")]
    public string weightsJsonPath;

    [Tooltip("The CarSensor whose live readings/layout this overlay visualizes.")]
    public CarSensor carSensor;

    public bool showOverlay = true;

    [Header("Panel (anchored to the bottom-right corner of the screen)")]
    public Vector2 panelSize = new Vector2(260, 260);
    public Vector2 panelAnchoredPosition = new Vector2(-20, 20);
    public float minDotRadius = 3f;
    public float maxDotRadius = 22f;

    [Header("Colors (mirrors the Python notebooks' TILE_COLORS palette)")]
    public Color colorSlippery = new Color(0.337f, 0.702f, 0.914f);      // #56b4e9
    public Color colorSpeedLimited = new Color(0.082f, 0.655f, 0.082f);  // #15a715
    public Color colorTerminal = new Color(0.839f, 0.153f, 0.157f);      // #d62728
    public Color colorAsphalt = new Color(0.30f, 0.30f, 0.30f);          // lightened from the
                                                                          // notebooks' near-black
                                                                          // #1d1d1d - stays visible
                                                                          // against the panel's own
                                                                          // dark translucent backing
    public Color colorGravel = new Color(0.627f, 0.604f, 0.561f);        // #a09a8f

    [Serializable]
    private class SensorWeightData
    {
        // Field names match export_sensor_weights.py's JSON keys exactly (snake_case) -
        // JsonUtility matches by exact name, not a configurable mapping.
        public string checkpoint;
        public int sensor_count;
        public int num_tile_types;
        public int hidden_units;
        public float max_value;
        public float[] values; // flat row-major: values[i * num_tile_types + h]
    }

    private SensorWeightData data;
    private GameObject canvasGO;
    private RectTransform[] dotTransforms;
    private Image[] dotImages;

    void Start()
    {
        if (carSensor == null)
        {
            Debug.LogError("SensorWeightOverlay: carSensor not assigned - disabling.", this);
            enabled = false;
            return;
        }
        if (!LoadWeights())
        {
            enabled = false;
            return;
        }
        BuildUI();
    }

    bool LoadWeights()
    {
        if (string.IsNullOrEmpty(weightsJsonPath) || !File.Exists(weightsJsonPath))
        {
            Debug.LogError($"SensorWeightOverlay: weightsJsonPath not found: '{weightsJsonPath}'. " +
                "Run export_sensor_weights.py against the checkpoint assigned to this Agent's " +
                "Behavior Parameters and point this field at the resulting .sensor_weights.json.", this);
            return false;
        }
        string json = File.ReadAllText(weightsJsonPath);
        data = JsonUtility.FromJson<SensorWeightData>(json);
        if (data == null || data.values == null || data.values.Length != data.sensor_count * data.num_tile_types)
        {
            Debug.LogError($"SensorWeightOverlay: failed to parse '{weightsJsonPath}', or its " +
                "values array doesn't match sensor_count*num_tile_types.", this);
            return false;
        }
        if (data.sensor_count != carSensor.SensorCount)
        {
            Debug.LogError($"SensorWeightOverlay: weights table has {data.sensor_count} sensors " +
                $"but carSensor currently reports {carSensor.SensorCount} - the checkpoint's fan " +
                "shape doesn't match this scene's current CarSensor Inspector settings (ring/point " +
                "counts changed since export?). Re-run export_sensor_weights.py against a matching " +
                "checkpoint, or fix the mismatch.", this);
            return false;
        }
        return true;
    }

    void BuildUI()
    {
        canvasGO = new GameObject("SensorWeightOverlayCanvas");
        canvasGO.transform.SetParent(transform, false);
        var canvas = canvasGO.AddComponent<Canvas>();
        canvas.renderMode = RenderMode.ScreenSpaceOverlay;
        canvas.sortingOrder = 1000; // draw on top of anything else in the scene's UI
        canvasGO.AddComponent<CanvasScaler>();
        canvasGO.AddComponent<GraphicRaycaster>();

        var panelGO = new GameObject("Panel");
        panelGO.transform.SetParent(canvasGO.transform, false);
        var panelRect = panelGO.AddComponent<RectTransform>();
        panelRect.anchorMin = new Vector2(1, 0);
        panelRect.anchorMax = new Vector2(1, 0);
        panelRect.pivot = new Vector2(1, 0);
        panelRect.sizeDelta = panelSize;
        panelRect.anchoredPosition = panelAnchoredPosition;
        var panelImage = panelGO.AddComponent<Image>();
        panelImage.color = new Color(0f, 0f, 0f, 0.55f);

        // Car icon at panel center, pointing "up" (forward) - orients the reader before they read
        // any dot positions.
        var carGO = new GameObject("CarIcon");
        carGO.transform.SetParent(panelGO.transform, false);
        var carRect = carGO.AddComponent<RectTransform>();
        carRect.anchorMin = carRect.anchorMax = new Vector2(0.5f, 0.5f);
        carRect.sizeDelta = new Vector2(10f, 16f);
        carRect.anchoredPosition = Vector2.zero;
        var carImage = carGO.AddComponent<Image>();
        carImage.color = Color.white;
        carImage.sprite = BuildTriangleSprite();

        var offsets = carSensor.SensorOffsets;
        int n = offsets.Count;
        dotTransforms = new RectTransform[n];
        dotImages = new Image[n];

        // Scale so the farthest sensor ring fits inside the panel with room for its own dot radius
        // - computed from the sensor's OWN current layout rather than a hardcoded ring distance, so
        // this stays correct if the Inspector's ring/point counts change later.
        float maxWorldRadius = 0.1f;
        for (int i = 0; i < n; i++)
            maxWorldRadius = Mathf.Max(maxWorldRadius, offsets[i].magnitude);
        float pixelsPerMeter = (Mathf.Min(panelSize.x, panelSize.y) / 2f - maxDotRadius) / maxWorldRadius;

        var circleSprite = BuildCircleSprite();
        for (int i = 0; i < n; i++)
        {
            var dotGO = new GameObject($"Sensor{i}");
            dotGO.transform.SetParent(panelGO.transform, false);
            var rect = dotGO.AddComponent<RectTransform>();
            rect.anchorMin = rect.anchorMax = new Vector2(0.5f, 0.5f);
            // offsets[i] = (local X = right, local Z = forward) - right maps straight to screen X;
            // forward maps to screen Y (up), matching the car icon's own up-pointing orientation.
            rect.anchoredPosition = new Vector2(offsets[i].x, offsets[i].y) * pixelsPerMeter;
            var image = dotGO.AddComponent<Image>();
            image.sprite = circleSprite;
            dotTransforms[i] = rect;
            dotImages[i] = image;
        }
    }

    void Update()
    {
        if (canvasGO != null && canvasGO.activeSelf != showOverlay)
            canvasGO.SetActive(showOverlay);
        if (!showOverlay || data == null || carSensor.readings == null || dotTransforms == null)
            return;

        int n = Mathf.Min(dotTransforms.Length, carSensor.readings.Length);
        for (int i = 0; i < n; i++)
        {
            int tileIndex = (int)carSensor.readings[i];
            float value = data.values[i * data.num_tile_types + tileIndex];
            float t = data.max_value > 0f ? Mathf.Clamp01(value / data.max_value) : 0f;
            float diameter = Mathf.Lerp(minDotRadius, maxDotRadius, t) * 2f;
            dotTransforms[i].sizeDelta = new Vector2(diameter, diameter);
            dotImages[i].color = ColorForTile((TileType)tileIndex);
        }
    }

    Color ColorForTile(TileType t)
    {
        switch (t)
        {
            case TileType.Slippery: return colorSlippery;
            case TileType.SpeedLimited: return colorSpeedLimited;
            case TileType.Terminal: return colorTerminal;
            case TileType.Asphalt: return colorAsphalt;
            case TileType.Gravel: return colorGravel;
            default: return Color.magenta; // should never hit - flags a TileType/color mapping gap
        }
    }

    static Sprite BuildCircleSprite()
    {
        const int size = 64;
        var tex = new Texture2D(size, size, TextureFormat.RGBA32, false);
        tex.wrapMode = TextureWrapMode.Clamp;
        var center = new Vector2(size / 2f, size / 2f);
        float radius = size / 2f;
        for (int y = 0; y < size; y++)
        {
            for (int x = 0; x < size; x++)
            {
                float d = Vector2.Distance(new Vector2(x + 0.5f, y + 0.5f), center);
                float alpha = Mathf.Clamp01(radius - d); // ~1px soft edge, filled disc inside
                tex.SetPixel(x, y, new Color(1f, 1f, 1f, alpha));
            }
        }
        tex.Apply();
        return Sprite.Create(tex, new Rect(0, 0, size, size), new Vector2(0.5f, 0.5f), size);
    }

    static Sprite BuildTriangleSprite()
    {
        const int size = 32;
        var tex = new Texture2D(size, size, TextureFormat.RGBA32, false);
        for (int y = 0; y < size; y++)
        {
            for (int x = 0; x < size; x++)
            {
                float t = y / (float)(size - 1); // 0 at bottom, 1 at top (apex)
                float halfWidthAtY = (size / 2f) * (1f - t);
                bool inside = Mathf.Abs(x - size / 2f) <= halfWidthAtY;
                tex.SetPixel(x, y, inside ? Color.white : new Color(1f, 1f, 1f, 0f));
            }
        }
        tex.Apply();
        return Sprite.Create(tex, new Rect(0, 0, size, size), new Vector2(0.5f, 0.5f), size);
    }
}
