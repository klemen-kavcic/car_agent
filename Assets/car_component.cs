using UnityEngine;

public class car_component : MonoBehaviour
{
    public Rigidbody rigid;
    public WheelCollider wheel1, wheel2, wheel3, wheel4;
    public float frontDriveSpeed, rearDriveSpeed;
    public GridManager gridManager;

    public float maxSteerAngle = 35f;
    public float maxSteerDeltaPerDecision = 3.5f;
    [Tooltip("Must match the DecisionRequester.DecisionPeriod on the Agent's GameObject, " +
        "since car_component has no ML-Agents dependency and can't read it directly.")]
    public int stepsPerDecision = 5;
    [Range(0f, 1f)] public float slipperyFrictionMultiplier = 0.05f;
    [Tooltip("Max magnitude (Newtons) of the random sideways disturbance force applied while on a Slippery tile — simulates losing control on ice, on top of the reduced wheel friction.")]
    public float slipperyDisturbanceForce = 400f;
    [Tooltip("Max magnitude (N·m) of the random yaw disturbance torque applied while on a Slippery tile.")]
    public float slipperyDisturbanceTorque = 200f;
    [Tooltip("How often (seconds) the random slippery-tile disturbance re-rolls to a new direction/magnitude. A short-lived push reads as a believable patch of ice rather than symmetric high-frequency jitter that cancels itself out.")]
    public float slipperyDisturbanceInterval = 0.4f;
    public float normalTopSpeedMS = 25f;       // measure in play mode, set here
    [Header("Optional vehicle-wide speed cap")]
    [Tooltip("When enabled, smoothly reduces drive torque near this speed. SpeedLimited and Gravel thresholds use this capped speed as their reference.")]
    public bool vehicleSpeedCapEnabled = false;
    [Min(0.1f)] public float vehicleSpeedCapMS = 14f;
    [Range(0.5f, 0.99f), Tooltip("The governor begins reducing drive torque at this fraction of vehicleSpeedCapMS.")]
    public float vehicleSpeedCapGovernorStartFraction = 0.90f;
    [Min(0f), Tooltip("Allowed overshoot (m/s) before the cap's catch-up brake engages.")]
    public float vehicleSpeedCapOverspeedMarginMS = 0.5f;
    [Tooltip("Modest brake torque (N·m per wheel), used only when the cap is exceeded by the overspeed margin.")]
    public float vehicleSpeedCapBrakeTorque = 900f;
    [Range(0f, 1f)] public float grassSpeedFraction = 0.10f;
    public float speedLimitBrakeTorque = 100000f;
    [Tooltip("Gravel tile top-speed cap, as a fraction of normalTopSpeedMS. Less restrictive than SpeedLimited/grassSpeedFraction - intended tunable range ~0.2-0.5.")]
    [Range(0f, 1f)] public float gravelSpeedFraction = 0.35f;
    [Tooltip("Gravel tile wheel friction multiplier. Only slightly reduced from full (1.0) grip, unlike Slippery's harsh slipperyFrictionMultiplier - no random disturbance force/torque on gravel.")]
    [Range(0f, 1f)] public float gravelFrictionMultiplier = 0.85f;
    [Range(0f, 1f)] public float reverseSpeedFraction = 0.15f;

    public float steerDeltaInput;
    [Range(0f, 1f)] public float accelerationInput;
    [Range(0f, 1f)] public float brakeInput;
    [Tooltip("Brake bias: normal (pedal) brake torque (N·m) applied to the front vs rear wheels.")]
    public float frontMaxBrakeTorque = 2200f;
    public float rearMaxBrakeTorque = 1000f;
    [Tooltip("Regenerative braking torque (N·m) applied to all wheels whenever the accelerator is fully released.")]
    public float regenBrakeTorque = 450f;
    [Tooltip("Gear can only be toggled when speed is at or below this (m/s).")]
    public float reverseEngageSpeed = 0.5f;
    public bool reverseGear; // true = wheels driven backward; flip with ToggleReverseGear()

    [HideInInspector] public float currentSteerAngle = 0f;
    public float currentSpeedMS; // read-only display, shows live speed in Inspector during play
    public TileType currentTileType = TileType.Asphalt;
    public bool regenActive; // read-only display, true while regen braking is currently applied
    public bool vehicleSpeedCapActive; // read-only display

    private float defaultSidewaysStiffness;
    private float defaultForwardStiffness;

    private float slipperyDisturbanceTimer;
    private float currentDisturbanceForce;
    private float currentDisturbanceTorque;

    // Tile limits are fractions of the speed that is actually available to this experiment.
    // With the optional cap disabled this is exactly normalTopSpeedMS, preserving old runs.
    public float EffectiveTopSpeedMS => vehicleSpeedCapEnabled
        ? Mathf.Min(normalTopSpeedMS, vehicleSpeedCapMS)
        : normalTopSpeedMS;

    // -1 means this surface has no explicit speed target (Asphalt/Slippery).
    public float CurrentTileTargetSpeedMS => currentTileType switch
    {
        TileType.SpeedLimited => EffectiveTopSpeedMS * grassSpeedFraction,
        TileType.Gravel       => EffectiveTopSpeedMS * gravelSpeedFraction,
        _                     => -1f,
    };

    // Called by CarAgent.OnEpisodeBegin() - these are private, so it can't reset them directly.
    // Forces a fresh disturbance roll next time the car is on ice, instead of possibly reusing a
    // stale force/torque left over from whatever the previous episode last rolled.
    public void ResetSlipperyDisturbance()
    {
        slipperyDisturbanceTimer = 0f;
        currentDisturbanceForce = 0f;
        currentDisturbanceTorque = 0f;
    }

    [Tooltip("Centre of mass offset from Rigidbody origin. Lower Y = more stable, less likely to tip.")]
    public Vector3 centreOfMassOffset = new Vector3(0f, -0.5f, 0f);
    public bool showCentreOfMassGizmo = true;

    void Start()
    {
        defaultSidewaysStiffness = wheel1.sidewaysFriction.stiffness;
        defaultForwardStiffness = wheel1.forwardFriction.stiffness;
        rigid.centerOfMass = centreOfMassOffset;
    }

    void FixedUpdate()
    {
        if (gridManager != null)
            currentTileType = gridManager.GetTileAt(transform.position);

        currentSpeedMS = rigid.linearVelocity.magnitude;

        float gearDirection = reverseGear ? -1f : 1f;
        float gearSpeedFraction = reverseGear ? reverseSpeedFraction : 1f;
        float frontMotor = frontDriveSpeed * accelerationInput * gearDirection * gearSpeedFraction;
        float rearMotor = rearDriveSpeed * accelerationInput * gearDirection * gearSpeedFraction;

        float steerDeltaThisStep = steerDeltaInput * maxSteerDeltaPerDecision / stepsPerDecision;
        currentSteerAngle = Mathf.Clamp(currentSteerAngle + steerDeltaThisStep, -maxSteerAngle, maxSteerAngle);

        ApplyTileEffects(frontMotor, rearMotor);

        wheel1.steerAngle = currentSteerAngle;
        wheel2.steerAngle = currentSteerAngle;
    }

    public void ToggleReverseGear()
    {
        if (CanShiftGear())
            reverseGear = !reverseGear;
    }

    // For a sustained "desired gear" control (e.g. an RL action) rather than a toggle event:
    // actual gear only ever catches up to the request once the gate opens, otherwise it holds.
    public void RequestReverseGear(bool desiredReverse)
    {
        if (CanShiftGear())
            reverseGear = desiredReverse;
    }

    bool CanShiftGear()
    {
        return currentSpeedMS <= reverseEngageSpeed;
    }

    void OnDrawGizmos()
    {
        if (rigid == null || !showCentreOfMassGizmo) return;
        Vector3 comWorldPos = Application.isPlaying
            ? rigid.worldCenterOfMass
            : rigid.transform.TransformPoint(centreOfMassOffset);
        Gizmos.color = Color.yellow;
        Gizmos.DrawSphere(comWorldPos, 0.2f);
    }

    void ApplyTileEffects(float frontMotor, float rearMotor)
    {
        float sidewaysStiffness = defaultSidewaysStiffness;
        float forwardStiffness = defaultForwardStiffness;
        float tileBrakeTorque = 0f;
        float governorStartSpeed = vehicleSpeedCapMS * vehicleSpeedCapGovernorStartFraction;
        vehicleSpeedCapActive = vehicleSpeedCapEnabled && currentSpeedMS >= governorStartSpeed;
        if (vehicleSpeedCapActive)
        {
            // Electronic-governor behaviour: taper propulsion before the cap, then remove it
            // at/above the cap. This avoids using wheel lock as the normal speed-control method.
            float t = Mathf.InverseLerp(governorStartSpeed, vehicleSpeedCapMS, currentSpeedMS);
            float driveScale = 1f - (t * t * (3f - 2f * t)); // smoothstep falloff
            frontMotor *= driveScale;
            rearMotor *= driveScale;
        }
        float vehicleCapBrakeTorque =
            vehicleSpeedCapEnabled && currentSpeedMS > vehicleSpeedCapMS + vehicleSpeedCapOverspeedMarginMS
                ? vehicleSpeedCapBrakeTorque
                : 0f;
        float tileSpeedReferenceMS = EffectiveTopSpeedMS;
        float tileTargetSpeedMS = float.PositiveInfinity;

        switch (currentTileType)
        {
            case TileType.Slippery:
                sidewaysStiffness *= slipperyFrictionMultiplier;
                forwardStiffness *= slipperyFrictionMultiplier;
                ApplySlipperyDisturbance();
                break;
            case TileType.SpeedLimited:
                tileTargetSpeedMS = tileSpeedReferenceMS * grassSpeedFraction;
                if (rigid.linearVelocity.magnitude > tileTargetSpeedMS)
                    tileBrakeTorque = speedLimitBrakeTorque;
                break;
            case TileType.Gravel:
                sidewaysStiffness *= gravelFrictionMultiplier;
                forwardStiffness *= gravelFrictionMultiplier;
                tileTargetSpeedMS = tileSpeedReferenceMS * gravelSpeedFraction;
                if (rigid.linearVelocity.magnitude > tileTargetSpeedMS)
                    tileBrakeTorque = speedLimitBrakeTorque;
                break;
            // TileType.Asphalt falls through with no special case - full default friction/speed,
            // the "safe default" surface under both Perlin and Voronoi (see GridManager.SpawnableTileType).
        }

        if (!float.IsPositiveInfinity(tileTargetSpeedMS))
        {
            // A tile brakes a car that enters too quickly. Near its own target speed, however,
            // taper propulsion instead of continuously fighting full motor torque with braking.
            float tileGovernorStart = tileTargetSpeedMS * vehicleSpeedCapGovernorStartFraction;
            if (currentSpeedMS >= tileGovernorStart)
            {
                float t = Mathf.InverseLerp(tileGovernorStart, tileTargetSpeedMS, currentSpeedMS);
                float driveScale = 1f - (t * t * (3f - 2f * t));
                frontMotor *= driveScale;
                rearMotor *= driveScale;
            }
        }

        regenActive = accelerationInput <= 0.01f;
        float regenTorque = regenActive ? regenBrakeTorque : 0f;
        float frontBrakeTorque = Mathf.Max(tileBrakeTorque, vehicleCapBrakeTorque, brakeInput * frontMaxBrakeTorque, regenTorque);
        float rearBrakeTorque = Mathf.Max(tileBrakeTorque, vehicleCapBrakeTorque, brakeInput * rearMaxBrakeTorque, regenTorque);

        ApplyToWheel(wheel1, sidewaysStiffness, forwardStiffness, frontMotor, frontBrakeTorque);
        ApplyToWheel(wheel2, sidewaysStiffness, forwardStiffness, frontMotor, frontBrakeTorque);
        ApplyToWheel(wheel3, sidewaysStiffness, forwardStiffness, rearMotor, rearBrakeTorque);
        ApplyToWheel(wheel4, sidewaysStiffness, forwardStiffness, rearMotor, rearBrakeTorque);
    }

    // Re-rolls a random sideways force + yaw torque every slipperyDisturbanceInterval seconds and
    // applies it continuously while on ice, so losing control reads as hitting a patch that pushes
    // you a consistent way for a bit, rather than symmetric per-step noise that averages to nothing.
    void ApplySlipperyDisturbance()
    {
        slipperyDisturbanceTimer -= Time.fixedDeltaTime;
        if (slipperyDisturbanceTimer <= 0f)
        {
            slipperyDisturbanceTimer = slipperyDisturbanceInterval;
            currentDisturbanceForce = Random.Range(-slipperyDisturbanceForce, slipperyDisturbanceForce);
            currentDisturbanceTorque = Random.Range(-slipperyDisturbanceTorque, slipperyDisturbanceTorque);
        }
        rigid.AddForce(transform.right * currentDisturbanceForce, ForceMode.Force);
        rigid.AddTorque(Vector3.up * currentDisturbanceTorque, ForceMode.Force);
    }

    void ApplyToWheel(WheelCollider w, float sidewaysStiffness, float forwardStiffness, float motor, float brakeTorque)
    {
        var sf = w.sidewaysFriction;
        sf.stiffness = sidewaysStiffness;
        w.sidewaysFriction = sf;

        var ff = w.forwardFriction;
        ff.stiffness = forwardStiffness;
        w.forwardFriction = ff;

        w.motorTorque = motor;
        w.brakeTorque = brakeTorque;
    }
}
