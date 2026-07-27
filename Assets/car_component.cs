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
    [Range(0f, 1f)] public float grassSpeedFraction = 0.10f;
    public float speedLimitBrakeTorque = 100000f;
    [Range(0f, 1f)] public float reverseSpeedFraction = 0.15f;

    public float steerDeltaInput;
    [Range(0f, 1f)] public float gasInput;
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
    public TileType currentTileType = TileType.Normal;
    public bool regenActive; // read-only display, true while regen braking is currently applied

    private float defaultSidewaysStiffness;
    private float defaultForwardStiffness;

    private float slipperyDisturbanceTimer;
    private float currentDisturbanceForce;
    private float currentDisturbanceTorque;

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
        float frontMotor = frontDriveSpeed * gasInput * gearDirection * gearSpeedFraction;
        float rearMotor = rearDriveSpeed * gasInput * gearDirection * gearSpeedFraction;

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

        switch (currentTileType)
        {
            case TileType.Slippery:
                sidewaysStiffness *= slipperyFrictionMultiplier;
                forwardStiffness *= slipperyFrictionMultiplier;
                ApplySlipperyDisturbance();
                break;
            case TileType.SpeedLimited:
                if (rigid.linearVelocity.magnitude > normalTopSpeedMS * grassSpeedFraction)
                    tileBrakeTorque = speedLimitBrakeTorque;
                break;
        }

        regenActive = gasInput <= 0.01f;
        float regenTorque = regenActive ? regenBrakeTorque : 0f;
        float frontBrakeTorque = Mathf.Max(tileBrakeTorque, brakeInput * frontMaxBrakeTorque, regenTorque);
        float rearBrakeTorque = Mathf.Max(tileBrakeTorque, brakeInput * rearMaxBrakeTorque, regenTorque);

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
