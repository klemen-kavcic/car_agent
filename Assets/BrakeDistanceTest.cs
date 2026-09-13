using System.Collections;
using UnityEngine;
using UnityEngine.InputSystem;
using Unity.MLAgents;

// Play-mode-only, repeatable asphalt braking test. It deliberately replaces the current runtime
// map with all Asphalt, then launches the car straight at an exact speed and measures projected
// forward distance until it is effectively stopped. Exiting Play mode restores the authored scene.
public class BrakeDistanceTest : MonoBehaviour
{
    public CarAgent agent;
    public car_component carController;
    public GridManager gridManager;

    [Header("Test setup")]
    [Min(0.1f)] public float startSpeedMS = 14f;
    [Min(0.001f)] public float stoppedSpeedMS = 0.05f;
    [Min(1f)] public float timeoutSeconds = 20f;

    [Header("Latest result (read-only)")]
    public string lastResult;
    public float lastStoppingDistanceM;
    public float lastStoppingTimeS;
    public bool testRunning;

    private DecisionRequester decisionRequester;

    void Awake()
    {
        if (agent == null) agent = GetComponent<CarAgent>();
        if (carController == null && agent != null) carController = agent.carController;
        if (gridManager == null && carController != null) gridManager = carController.gridManager;
        decisionRequester = GetComponent<DecisionRequester>();
    }

    void Update()
    {
        Keyboard keyboard = Keyboard.current;
        if (keyboard == null || testRunning) return;

        // T = normal full pedal brake. Y = accelerator release, therefore regen-only braking.
        if (keyboard.tKey.wasPressedThisFrame) StartCoroutine(RunTest(fullPedalBrake: true));
        if (keyboard.yKey.wasPressedThisFrame) StartCoroutine(RunTest(fullPedalBrake: false));
    }

    [ContextMenu("Run 14 m/s full-pedal brake test")]
    public void RunFullPedalBrakeTest()
    {
        if (!testRunning) StartCoroutine(RunTest(fullPedalBrake: true));
    }

    [ContextMenu("Run 14 m/s release / regen-only brake test")]
    public void RunRegenOnlyBrakeTest()
    {
        if (!testRunning) StartCoroutine(RunTest(fullPedalBrake: false));
    }

    IEnumerator RunTest(bool fullPedalBrake)
    {
        if (agent == null || carController == null || gridManager == null || carController.rigid == null)
        {
            Debug.LogError("BrakeDistanceTest: assign Agent, car controller, GridManager and Rigidbody.", this);
            yield break;
        }

        testRunning = true;
        bool agentWasEnabled = agent.enabled;
        bool requesterWasEnabled = decisionRequester != null && decisionRequester.enabled;
        float oldAcceleration = carController.accelerationInput;
        float oldBrake = carController.brakeInput;
        bool oldForceAllNormal = gridManager.forceAllNormal;

        // The Agent must not overwrite the test inputs every decision step.
        agent.enabled = false;
        if (decisionRequester != null) decisionRequester.enabled = false;
        gridManager.forceAllNormal = true;
        gridManager.Regenerate();
        // Destroy() of the old tile objects completes at frame end; wait so no old/new collider
        // overlap remains when the wheel contacts settle.
        yield return null;

        Rigidbody body = carController.rigid;
        Vector3 start = gridManager.GridOrigin + new Vector3(
            (gridManager.gridSize / 2 + 0.5f) * gridManager.cellSize,
            transform.position.y,
            8.5f * gridManager.cellSize);
        transform.SetPositionAndRotation(start, Quaternion.identity); // forward is +Z
        body.linearVelocity = Vector3.zero;
        body.angularVelocity = Vector3.zero;
        carController.currentSteerAngle = 0f;
        carController.reverseGear = false;
        carController.accelerationInput = 0f;
        carController.brakeInput = 0f;
        Physics.SyncTransforms();

        // Let wheel contacts settle for one fixed tick before injecting the precisely known speed.
        yield return new WaitForFixedUpdate();
        Vector3 forward = transform.forward;
        Vector3 positionAtRelease = transform.position;
        body.linearVelocity = forward * startSpeedMS;
        carController.accelerationInput = 0f;
        carController.brakeInput = fullPedalBrake ? 1f : 0f;

        float elapsed = 0f;
        while (elapsed < timeoutSeconds)
        {
            // Hold the requested brake state: no motor torque, with either pedal brake or the
            // car component's usual regen-on-release behaviour.
            carController.accelerationInput = 0f;
            carController.brakeInput = fullPedalBrake ? 1f : 0f;
            yield return new WaitForFixedUpdate();
            elapsed += Time.fixedDeltaTime;
            if (body.linearVelocity.magnitude <= stoppedSpeedMS) break;
        }

        lastStoppingDistanceM = Mathf.Max(0f, Vector3.Dot(transform.position - positionAtRelease, forward));
        lastStoppingTimeS = elapsed;
        bool timedOut = elapsed >= timeoutSeconds;
        string mode = fullPedalBrake ? "full pedal brake" : "release / regen-only";
        lastResult = timedOut
            ? $"{mode}: TIMEOUT after {elapsed:F2}s, distance {lastStoppingDistanceM:F2}m"
            : $"{mode}: {startSpeedMS:F1} -> {stoppedSpeedMS:F2} m/s in {lastStoppingDistanceM:F2}m / {elapsed:F2}s";
        Debug.Log($"[BrakeDistanceTest] {lastResult}", this);

        body.linearVelocity = Vector3.zero;
        carController.accelerationInput = oldAcceleration;
        carController.brakeInput = oldBrake;
        gridManager.forceAllNormal = oldForceAllNormal;
        agent.enabled = agentWasEnabled;
        if (decisionRequester != null) decisionRequester.enabled = requesterWasEnabled;
        testRunning = false;
    }
}
