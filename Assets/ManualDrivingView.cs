using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.UI;

// Toggleable local-driving view for testing what a human can do with precisely the policy's
// information. It switches off the map overview, follows the car from behind, and displays the
// fixed observations plus every laser's decoded five-value input. It does not alter the agent,
// physics or observation vector.
public class ManualDrivingView : MonoBehaviour
{
    [Header("References")]
    public CarAgent agent;
    public car_component carController;
    public Camera overviewCamera;
    public Camera drivingCamera;

    [Header("Toggle")]
    [Tooltip("Enable the close driving camera and input dashboard. F1 toggles this during Play mode.")]
    public bool manualDrivingViewEnabled;
    [Tooltip("Show the eight laser rays around the car while the manual driving view is active. This is visual-only.")]
    public bool showWorldLaserRaysInManualView = true;
    [Tooltip("Hide all tile renderers in manual view, so navigation uses only the observation HUD and laser rays. The goal and car remain visible.")]
    public bool hideMapTilesInManualView = true;

    [Header("Steering indicator")]
    [Tooltip("Draw bright direction lines above the two front wheels while using manual view.")]
    public bool showWheelSteeringIndicator = true;
    [Min(0.1f)] public float steeringIndicatorLength = 2.2f;
    [Min(0.01f)] public float steeringIndicatorWidth = 0.09f;
    [Min(0f)] public float steeringIndicatorHeight = 0.28f;

    [Header("Third-person camera")]
    public Vector3 localFollowOffset = new Vector3(0f, 4.5f, -7.5f);
    [Min(0f)] public float lookAheadMeters = 7f;
    [Min(0f)] public float cameraPositionSmoothTime = 0.15f;
    [Range(0f, 20f)] public float cameraRotationSharpness = 10f;

    private bool appliedManualState;
    private Vector3 followVelocity;
    private GameObject hudRoot;
    private Text hudText;
    private GridManager gridManager;
    private GameObject steeringIndicatorRoot;
    private LineRenderer[] steeringLines;
    private Material steeringMaterial;

    void Awake()
    {
        if (agent == null) agent = GetComponent<CarAgent>();
        if (carController == null && agent != null) carController = agent.carController;
        if (agent != null && agent.laserTileSensor != null)
            gridManager = agent.laserTileSensor.gridManager;
        if (gridManager == null && carController != null)
            gridManager = carController.gridManager;
    }

    void Start()
    {
        ApplyView(manualDrivingViewEnabled);
    }

    void Update()
    {
        Keyboard keyboard = Keyboard.current;
        if (keyboard != null && keyboard.f1Key.wasPressedThisFrame)
            manualDrivingViewEnabled = !manualDrivingViewEnabled;

        if (manualDrivingViewEnabled != appliedManualState)
            ApplyView(manualDrivingViewEnabled);

        if (manualDrivingViewEnabled && hudText != null)
            UpdateHud();
    }

    void LateUpdate()
    {
        if (!manualDrivingViewEnabled || drivingCamera == null || agent == null) return;

        Transform car = agent.transform;
        Vector3 desiredPosition = car.TransformPoint(localFollowOffset);
        drivingCamera.transform.position = Vector3.SmoothDamp(
            drivingCamera.transform.position, desiredPosition, ref followVelocity, cameraPositionSmoothTime);

        Vector3 target = car.position + car.forward * lookAheadMeters + Vector3.up * 0.8f;
        Vector3 toTarget = target - drivingCamera.transform.position;
        if (toTarget.sqrMagnitude > 0.0001f)
        {
            Quaternion desiredRotation = Quaternion.LookRotation(toTarget.normalized, Vector3.up);
            float t = 1f - Mathf.Exp(-cameraRotationSharpness * Time.deltaTime);
            drivingCamera.transform.rotation = Quaternion.Slerp(drivingCamera.transform.rotation, desiredRotation, t);
        }

        UpdateSteeringIndicator();
    }

    public void SetManualDrivingView(bool enabled)
    {
        manualDrivingViewEnabled = enabled;
        ApplyView(enabled);
    }

    void ApplyView(bool enabled)
    {
        appliedManualState = enabled;
        SetCameraEnabled(overviewCamera, !enabled);
        SetCameraEnabled(drivingCamera, enabled);
        if (agent != null && agent.laserTileSensor != null)
            agent.laserTileSensor.SetRuntimeLasersForcedVisible(enabled && showWorldLaserRaysInManualView);
        if (gridManager != null)
            gridManager.SetTileVisualsVisible(!enabled || !hideMapTilesInManualView);
        if (!enabled && steeringIndicatorRoot != null)
            steeringIndicatorRoot.SetActive(false);

        if (enabled)
        {
            EnsureHud();
            UpdateHud();
        }
        if (hudRoot != null) hudRoot.SetActive(enabled);
    }

    void UpdateSteeringIndicator()
    {
        if (!manualDrivingViewEnabled || !showWheelSteeringIndicator || carController == null ||
            carController.wheel1 == null || carController.wheel2 == null)
        {
            if (steeringIndicatorRoot != null) steeringIndicatorRoot.SetActive(false);
            return;
        }

        EnsureSteeringIndicator();
        if (steeringIndicatorRoot == null) return;
        steeringIndicatorRoot.SetActive(true);

        Vector3 direction = Quaternion.AngleAxis(carController.currentSteerAngle, agent.transform.up) * agent.transform.forward;
        WheelCollider[] frontWheels = { carController.wheel1, carController.wheel2 };
        for (int i = 0; i < frontWheels.Length; i++)
        {
            Vector3 start = frontWheels[i].transform.position + Vector3.up * steeringIndicatorHeight;
            steeringLines[i].widthMultiplier = steeringIndicatorWidth;
            steeringLines[i].SetPosition(0, start);
            steeringLines[i].SetPosition(1, start + direction.normalized * steeringIndicatorLength);
        }
    }

    void EnsureSteeringIndicator()
    {
        if (steeringIndicatorRoot != null) return;

        Shader shader = Shader.Find("Universal Render Pipeline/Unlit");
        if (shader == null) shader = Shader.Find("Sprites/Default");
        if (shader == null)
        {
            Debug.LogError("ManualDrivingView: no unlit shader available for steering indicator.", this);
            return;
        }

        steeringIndicatorRoot = new GameObject("FrontWheelSteeringIndicator");
        steeringIndicatorRoot.transform.SetParent(agent.transform, false);
        steeringMaterial = new Material(shader);
        steeringLines = new LineRenderer[2];
        for (int i = 0; i < steeringLines.Length; i++)
        {
            GameObject lineGO = new GameObject($"FrontWheelDirection{i}");
            lineGO.transform.SetParent(steeringIndicatorRoot.transform, false);
            LineRenderer line = lineGO.AddComponent<LineRenderer>();
            line.material = steeringMaterial;
            line.positionCount = 2;
            line.useWorldSpace = true;
            line.alignment = LineAlignment.View;
            line.numCapVertices = 3;
            line.startColor = Color.yellow;
            line.endColor = Color.yellow;
            steeringLines[i] = line;
        }
    }

    void OnDestroy()
    {
        if (steeringMaterial == null) return;
        if (Application.isPlaying) Destroy(steeringMaterial);
        else DestroyImmediate(steeringMaterial);
    }

    static void SetCameraEnabled(Camera cameraToSet, bool enabled)
    {
        if (cameraToSet == null) return;
        cameraToSet.enabled = enabled;
        AudioListener listener = cameraToSet.GetComponent<AudioListener>();
        if (listener != null) listener.enabled = enabled;
    }

    void EnsureHud()
    {
        if (hudRoot != null) return;

        hudRoot = new GameObject("ManualDrivingObservationHUD");
        Canvas canvas = hudRoot.AddComponent<Canvas>();
        canvas.renderMode = RenderMode.ScreenSpaceOverlay;
        canvas.sortingOrder = 999; // beneath the laser panel (1000), above the world
        hudRoot.AddComponent<CanvasScaler>();

        GameObject panel = new GameObject("ObservationPanel");
        panel.transform.SetParent(hudRoot.transform, false);
        RectTransform panelRect = panel.AddComponent<RectTransform>();
        panelRect.anchorMin = new Vector2(0f, 1f);
        panelRect.anchorMax = new Vector2(0f, 1f);
        panelRect.pivot = new Vector2(0f, 1f);
        panelRect.anchoredPosition = new Vector2(18f, -18f);
        panelRect.sizeDelta = new Vector2(500f, 510f);
        Image panelImage = panel.AddComponent<Image>();
        panelImage.color = new Color(0f, 0f, 0f, 0.62f);

        GameObject textGO = new GameObject("ObservationText");
        textGO.transform.SetParent(panel.transform, false);
        RectTransform textRect = textGO.AddComponent<RectTransform>();
        textRect.anchorMin = Vector2.zero;
        textRect.anchorMax = Vector2.one;
        textRect.offsetMin = new Vector2(12f, 8f);
        textRect.offsetMax = new Vector2(-12f, -8f);
        hudText = textGO.AddComponent<Text>();
        hudText.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
        hudText.fontSize = 14;
        hudText.alignment = TextAnchor.UpperLeft;
        hudText.horizontalOverflow = HorizontalWrapMode.Overflow;
        hudText.verticalOverflow = VerticalWrapMode.Overflow;
        hudText.color = Color.white;
    }

    void UpdateHud()
    {
        if (agent == null || carController == null || hudText == null) return;

        Rigidbody rb = agent.rb;
        Vector3 toGoal = agent.goal.position - agent.transform.position;
        Vector3 goalDirection = agent.transform.InverseTransformDirection(toGoal.normalized);
        Vector3 localVelocity = rb != null
            ? agent.transform.InverseTransformDirection(rb.linearVelocity)
            : Vector3.zero;
        float yawRate = rb != null ? agent.transform.InverseTransformDirection(rb.angularVelocity).y : 0f;
        float steerNormalized = carController.maxSteerAngle > 0f
            ? carController.currentSteerAngle / carController.maxSteerAngle : 0f;
        float pedal = carController.accelerationInput + carController.brakeInput;
        bool braking = carController.brakeInput > 0f;

        System.Text.StringBuilder sb = new System.Text.StringBuilder(900);
        sb.AppendLine("MANUAL DRIVING VIEW  —  F1: overview camera");
        sb.AppendLine("Controls: W accelerate | S brake | A/D steer | Q reverse | N reset");
        sb.AppendLine();
        sb.AppendFormat("Goal local normalized: ({0:F2}, {1:F2}, {2:F2})   distance: {3:F1} m\n",
            goalDirection.x, goalDirection.y, goalDirection.z, toGoal.magnitude);
        sb.AppendFormat("Velocity local: ({0:F2}, {1:F2}, {2:F2}) m/s   yaw: {3:F2} rad/s\n",
            localVelocity.x, localVelocity.y, localVelocity.z, yawRate);
        sb.AppendFormat("Steer: {0:F2} ({1:F1}\u00b0)   gear: {2}   pedal: {3} {4:F2}\n",
            steerNormalized, carController.currentSteerAngle, carController.reverseGear ? "reverse" : "forward",
            braking ? "brake" : "accelerate", pedal);
        sb.AppendFormat("Current tile: {0}  one-hot {1}\n", carController.currentTileType,
            TileOneHotText(carController.currentTileType));
        if (agent.includeRemainingStepsObservation)
            sb.AppendFormat("Remaining steps: {0:F3}\n", agent.MaxStep > 0 ?
                Mathf.Clamp01(1f - (float)agent.StepCount / agent.MaxStep) : 1f);

        if (agent.tileObservationMode == CarAgent.TileObservationMode.ContinuousLasers &&
            agent.laserTileSensor != null && agent.laserTileSensor.gridManager != null)
        {
            sb.AppendLine("Lasers — special type one-hot | special distance | road boundary:");
            LaserTileSensor laser = agent.laserTileSensor;
            for (int i = 0; i < LaserTileSensor.DirectionCount; i++)
            {
                laser.Trace(laser.transform.position,
                    laser.transform.TransformDirection(LaserTileSensor.GetLocalDirection(i)),
                    out TileType special, out float specialDistance, out float roadDistance);
                sb.AppendFormat(" {0,-2}: {1,-12} {2}  {3,5:F1} m  {4,5:F1} m\n",
                    DirectionName(i), special, SpecialOneHotText(special), specialDistance, roadDistance);
            }
        }
        else
        {
            sb.AppendLine("Fan mode: the lower-right panel shows the fan layout.");
        }
        hudText.text = sb.ToString();
    }

    static string TileOneHotText(TileType tile)
    {
        return tile switch
        {
            TileType.Slippery => "[1 0 0 0 0]",
            TileType.SpeedLimited => "[0 1 0 0 0]",
            TileType.Terminal => "[0 0 1 0 0]",
            TileType.Asphalt => "[0 0 0 1 0]",
            TileType.Gravel => "[0 0 0 0 1]",
            _ => "[? ? ? ? ?]",
        };
    }

    static string SpecialOneHotText(TileType tile)
    {
        return tile switch
        {
            TileType.Slippery => "[1 0 0]",
            TileType.SpeedLimited => "[0 1 0]",
            TileType.Terminal => "[0 0 1]",
            _ => "[0 0 0]",
        };
    }

    static string DirectionName(int index)
    {
        string[] names = { "F", "FR", "R", "BR", "B", "BL", "L", "FL" };
        return names[index];
    }
}
