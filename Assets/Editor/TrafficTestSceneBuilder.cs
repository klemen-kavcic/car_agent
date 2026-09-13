using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using Unity.MLAgents.Policies;

public static class TrafficTestSceneBuilder
{
    private const string SourceScene = "Assets/Scenes/SampleScene.unity";
    private const string TrafficScene = "Assets/Scenes/TrafficTwoCar.unity";

    [MenuItem("Tools/Traffic/Create two-car laser test scene")]
    public static void CreateScene()
    {
        if (File.Exists(TrafficScene))
        {
            Debug.LogWarning("Traffic scene already exists; refusing to overwrite it: " + TrafficScene);
            return;
        }
        if (!EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo()) return;
        if (!AssetDatabase.CopyAsset(SourceScene, TrafficScene))
        {
            Debug.LogError("Could not copy " + SourceScene);
            return;
        }
        var scene = EditorSceneManager.OpenScene(TrafficScene, OpenSceneMode.Single);
        CarAgent[] existingCars = Object.FindObjectsByType<CarAgent>(FindObjectsSortMode.None);
        GridManager grid = Object.FindFirstObjectByType<GridManager>();
        TextAsset manifest = AssetDatabase.LoadAssetAtPath<TextAsset>(
            "Assets/Resources/Traffic/traffic_pairs.json");
        if (existingCars.Length != 1 || grid == null || manifest == null)
        {
            Debug.LogError("Expected one source car, one grid and traffic_pairs.json in the copied scene.");
            return;
        }

        CarAgent car0 = existingCars[0];
        GameObject duplicate = Object.Instantiate(car0.gameObject);
        duplicate.name = "TrafficCar1";
        CarAgent car1 = duplicate.GetComponent<CarAgent>();
        GameObject goal1 = Object.Instantiate(car0.goal.gameObject);
        goal1.name = "TrafficGoal1";
        car1.goal = goal1.transform;
        GameObject spawn1 = Object.Instantiate(car0.spawnPoint.gameObject);
        spawn1.name = "TrafficSpawn1";
        car1.spawnPoint = spawn1.transform;

        // The original body collider is 1 x 1 m in the scene, while the wheelbase
        // spans roughly 3 m. Use this traffic-only body size for both physical
        // contact and the analytic laser hit test; do not change static tile rules.
        foreach (CarAgent car in new[] { car0, car1 })
        {
            car.tileObservationMode = CarAgent.TileObservationMode.ContinuousLasers;
            car.includeSensorObservations = true;
            car.disableCurriculumInEditor = true;
            car.MaxStep = 0;
            car.GetComponent<BoxCollider>().size = new Vector3(2.0f, 1.0f, 3.7f);
            car.carController.vehicleSpeedCapEnabled = true;
            car.carController.vehicleSpeedCapMS = 3f;
            car.carController.regenBrakeTorque = 450f;
            car.laserTileSensor.specialDistanceScaleM = 10f;
            car.laserTileSensor.roadBoundaryDistanceScaleM = 3f;
            car.GetComponent<BehaviorParameters>().BehaviorType = BehaviorType.Default;
        }
        car0.gameObject.name = "TrafficCar0";
        car0.GetComponent<BehaviorParameters>().BehaviorName = "TrafficCar0";
        car1.GetComponent<BehaviorParameters>().BehaviorName = "TrafficCar1";
        foreach (var component in duplicate.GetComponents<ManualDrivingView>()) component.enabled = false;
        foreach (var component in duplicate.GetComponents<SensorWeightOverlay>()) component.enabled = false;
        foreach (var component in duplicate.GetComponents<BrakeDistanceTest>()) component.enabled = false;

        var managerObject = new GameObject("TrafficScenarioManager");
        var manager = managerObject.AddComponent<TrafficScenarioManager>();
        manager.gridManager = grid;
        manager.cars = new[] { car0, car1 };
        manager.scenarioJson = manifest;
        manager.sharedMaxSteps = 5000;
        manager.scenarioIndex = 0;
        car0.trafficManager = manager;
        car0.trafficSeatIndex = 0;
        car1.trafficManager = manager;
        car1.trafficSeatIndex = 1;

        EditorSceneManager.MarkSceneDirty(scene);
        EditorSceneManager.SaveScene(scene);
        Debug.Log("Created " + TrafficScene + ". Build this scene for Python evaluation.");
    }
}
