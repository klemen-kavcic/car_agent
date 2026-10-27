using UnityEngine;

// Marker for a kinematic, parked traffic vehicle used by the recovery
// curriculum.  CarAgent and TrafficScenarioManager treat it like another car
// for collisions and vehicle-distance laser observations, but it is not an
// ML-Agent seat and never receives rewards or resets on its own.
public class TrafficVehicleObstacle : MonoBehaviour
{
}
