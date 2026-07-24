using UnityEngine;
 
public class wheel : MonoBehaviour
{
    public WheelCollider wheelCollider;
    public Transform wheelMesh;
    public bool wheelTurn;
    public float spinDirection = 1f;
 
    private float spinAngle;
    private Quaternion meshOffset;
 
    void Start()
    {
        meshOffset = wheelMesh.localRotation; // captures the baked mesh offset once
    }
 
    void Update()
    {
        // --- Suspension travel (NEW) ---
        // Without this, the mesh never follows the collider's up/down movement,
        // so the wheel spins/steers in place but ignores bumps, load, and droop.
        Vector3 pos;
        Quaternion colliderRot;
        wheelCollider.GetWorldPose(out pos, out colliderRot);
        wheelMesh.position = pos; // world-space assignment, safe regardless of parent chain
 
        // --- Spin + steer (unchanged from original) ---
        spinAngle += spinDirection * wheelCollider.rpm / 60f * 360f * Time.deltaTime;
 
        float steerAngle = wheelTurn ? wheelCollider.steerAngle : 0f;
 
        wheelMesh.localRotation =
            Quaternion.Euler(0f, 0f, steerAngle) *
            meshOffset *
            Quaternion.Euler(0f, 0f, spinAngle);
    }
}