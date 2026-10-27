using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using Unity.MLAgents.Policies;

public static class TrafficTestSceneBuilder
{
    private const string SourceScene = "Assets/Scenes/SampleScene.unity";
    private const string TrafficScene = "Assets/Scenes/TrafficTwoCar.unity";
    private const string FullValidationScene = "Assets/Scenes/TrafficAllValidation.unity";
    private const string TrainingScene = "Assets/Scenes/TrafficTraining.unity";
    private const string VehicleTrainingScene = "Assets/Scenes/TrafficTrainingVehicle.unity";
    private const string VehicleValidationScene = "Assets/Scenes/TrafficAllValidationVehicle.unity";
    private const string VehicleArcadeTrainingScene = "Assets/Scenes/TrafficTrainingVehicleArcade.unity";
    private const string VehicleArcadeValidationScene = "Assets/Scenes/TrafficAllValidationVehicleArcade.unity";
    private const string VehicleSignedArcadeTrainingScene = "Assets/Scenes/TrafficTrainingVehicleSignedArcade.unity";
    private const string VehicleSignedArcadeValidationScene = "Assets/Scenes/TrafficAllValidationVehicleSignedArcade.unity";
    private const string VehicleSignedArcadeTimeTrainingScene = "Assets/Scenes/TrafficTrainingVehicleSignedArcadeTimeRemaining.unity";
    private const string VehicleSignedArcadeTimeValidationScene = "Assets/Scenes/TrafficAllValidationVehicleSignedArcadeTimeRemaining.unity";
    private const string VehicleSignedArcadeTimeDenseTrainingScene = "Assets/Scenes/TrafficTrainingVehicleSignedArcadeTimeDenseLidar.unity";
    private const string VehicleSignedArcadeTimeDenseValidationScene = "Assets/Scenes/TrafficAllValidationVehicleSignedArcadeTimeDenseLidar.unity";
    private const string VehicleSignedArcadeFineTrainingScene = "Assets/Scenes/TrafficTrainingVehicleSignedArcadeFine330.unity";
    private const string VehicleSignedArcadeFineValidationScene = "Assets/Scenes/TrafficAllValidationVehicleSignedArcadeFine330.unity";
    private const string VehicleSignedArcadeTimeFineDenseTrainingScene = "Assets/Scenes/TrafficTrainingVehicleSignedArcadeTimeFine330DenseLidar.unity";
    private const string VehicleSignedArcadeTimeFineDenseValidationScene = "Assets/Scenes/TrafficAllValidationVehicleSignedArcadeTimeFine330DenseLidar.unity";
    private const string FullValidationManifest =
        "Assets/Resources/Traffic/traffic_validation_maps.json";
    private const string TrainingManifest =
        "Assets/Resources/Traffic/traffic_training_maps.json";

    [MenuItem("Tools/Traffic/Create vehicle-distance training scene")]
    public static void CreateVehicleTrainingScene()
    {
        CreateVehicleDistanceScene(TrainingScene, VehicleTrainingScene);
    }

    [MenuItem("Tools/Traffic/Create vehicle-distance validation scene")]
    public static void CreateVehicleValidationScene()
    {
        CreateVehicleDistanceScene(FullValidationScene, VehicleValidationScene);
    }

    [MenuItem("Tools/Traffic/Create vehicle-distance arcade training scene")]
    public static void CreateVehicleArcadeTrainingScene()
    {
        CreateVehicleArcadeScene(VehicleTrainingScene, VehicleArcadeTrainingScene, "training");
    }

    [MenuItem("Tools/Traffic/Create vehicle-distance arcade validation scene")]
    public static void CreateVehicleArcadeValidationScene()
    {
        CreateVehicleArcadeScene(VehicleValidationScene, VehicleArcadeValidationScene, "validation");
    }

    [MenuItem("Tools/Traffic/Create vehicle-distance signed-pedal arcade training scene")]
    public static void CreateVehicleSignedArcadeTrainingScene()
    {
        CreateVehicleSignedArcadeScene(VehicleTrainingScene, VehicleSignedArcadeTrainingScene, "training");
    }

    [MenuItem("Tools/Traffic/Create vehicle-distance signed-pedal arcade validation scene")]
    public static void CreateVehicleSignedArcadeValidationScene()
    {
        CreateVehicleSignedArcadeScene(VehicleValidationScene, VehicleSignedArcadeValidationScene, "validation");
    }

    [MenuItem("Tools/Traffic/Create signed-pedal remaining-time training scene")]
    public static void CreateVehicleSignedArcadeTimeTrainingScene()
    {
        CreateVehicleSignedArcadeTimeScene(
            VehicleSignedArcadeTrainingScene, VehicleSignedArcadeTimeTrainingScene, "training");
    }

    [MenuItem("Tools/Traffic/Create signed-pedal remaining-time validation scene")]
    public static void CreateVehicleSignedArcadeTimeValidationScene()
    {
        CreateVehicleSignedArcadeTimeScene(
            VehicleSignedArcadeValidationScene, VehicleSignedArcadeTimeValidationScene, "validation");
    }

    [MenuItem("Tools/Traffic/Create normal-map signed-pedal time/dense-lidar training scene")]
    public static void CreateVehicleSignedArcadeTimeDenseTrainingScene()
    {
        CreateVehicleSignedArcadeTimeDenseScene(
            VehicleSignedArcadeTimeTrainingScene, VehicleSignedArcadeTimeDenseTrainingScene, "training");
    }

    [MenuItem("Tools/Traffic/Create normal-map signed-pedal time/dense-lidar validation scene")]
    public static void CreateVehicleSignedArcadeTimeDenseValidationScene()
    {
        CreateVehicleSignedArcadeTimeDenseScene(
            VehicleSignedArcadeTimeValidationScene, VehicleSignedArcadeTimeDenseValidationScene, "validation");
    }

    // This changes only the sensor density.  In particular, it deliberately
    // preserves the established 66x66, 1.5m normal-map environment so its
    // performance can be compared directly with the 90-observation
    // time+goal-removal baseline.
    static void CreateVehicleSignedArcadeTimeDenseScene(string sourceScene, string destinationScene, string purpose)
    {
        if (File.Exists(destinationScene))
        {
            Debug.LogWarning("Normal-map time/dense-lidar " + purpose +
                             " scene already exists; refusing to overwrite it: " + destinationScene);
            return;
        }
        if (!File.Exists(sourceScene))
        {
            Debug.LogError("Create the signed-pedal remaining-time " + purpose + " scene first: " + sourceScene);
            return;
        }
        if (!EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo()) return;
        if (!AssetDatabase.CopyAsset(sourceScene, destinationScene))
        {
            Debug.LogError("Could not copy " + sourceScene);
            return;
        }

        var scene = EditorSceneManager.OpenScene(destinationScene, OpenSceneMode.Single);
        var manager = Object.FindFirstObjectByType<TrafficScenarioManager>();
        GridManager grid = manager != null ? manager.gridManager : Object.FindFirstObjectByType<GridManager>();
        if (manager == null || grid == null || manager.cars == null || manager.cars.Length != 10)
        {
            Debug.LogError("Expected a ten-car TrafficScenarioManager and GridManager in " + destinationScene);
            return;
        }
        if (grid.gridSize != 66 || !Mathf.Approximately(grid.cellSize, 1.5f) || grid.voronoiMapRefinement != 1)
        {
            Debug.LogError("The normal-map dense-lidar comparison must start from the 66x66, 1.5m scene.");
            return;
        }

        foreach (CarAgent car in manager.cars)
        {
            if (car == null || car.laserTileSensor == null || !car.includeRemainingStepsObservation)
            {
                Debug.LogError("Every traffic car needs LaserTileSensor and remaining-time observation.");
                return;
            }
            car.laserTileSensor.denseFullCircle15DegreeRays = true;
            var behavior = car.GetComponent<BehaviorParameters>();
            if (behavior == null)
            {
                Debug.LogError("Every traffic car needs BehaviorParameters.");
                return;
            }
            var serialized = new SerializedObject(behavior);
            serialized.FindProperty("m_BrainParameters")
                .FindPropertyRelative("VectorObservationSize").intValue =
                    17 + car.laserTileSensor.ActiveObservationCount + 1;
            serialized.ApplyModifiedPropertiesWithoutUndo();
        }
        EditorSceneManager.MarkSceneDirty(scene);
        EditorSceneManager.SaveScene(scene);
        Debug.Log("Created " + destinationScene + ": normal 66x66 maps, 24 rays at 15 degrees, " +
                  "vehicle distances, and remaining time = 162 observations.");
    }

    [MenuItem("Tools/Traffic/Create 5x refined signed-pedal training scene")]
    public static void CreateVehicleSignedArcadeFineTrainingScene()
    {
        CreateVehicleSignedArcadeFineScene(
            VehicleSignedArcadeTrainingScene, VehicleSignedArcadeFineTrainingScene, "training");
    }

    [MenuItem("Tools/Traffic/Create 5x refined signed-pedal validation scene")]
    public static void CreateVehicleSignedArcadeFineValidationScene()
    {
        CreateVehicleSignedArcadeFineScene(
            VehicleSignedArcadeValidationScene, VehicleSignedArcadeFineValidationScene, "validation");
    }

    [MenuItem("Tools/Traffic/Create 5x refined signed-pedal time/dense-lidar training scene")]
    public static void CreateVehicleSignedArcadeTimeFineDenseTrainingScene()
    {
        CreateVehicleSignedArcadeTimeFineDenseScene(
            VehicleSignedArcadeTimeTrainingScene, VehicleSignedArcadeTimeFineDenseTrainingScene, "training");
    }

    [MenuItem("Tools/Traffic/Create 5x refined signed-pedal time/dense-lidar validation scene")]
    public static void CreateVehicleSignedArcadeTimeFineDenseValidationScene()
    {
        CreateVehicleSignedArcadeTimeFineDenseScene(
            VehicleSignedArcadeTimeValidationScene, VehicleSignedArcadeTimeFineDenseValidationScene, "validation");
    }

    // A deliberately separate 162-observation scene: it keeps the established
    // 89/90-observation scenes and ONNX models valid while testing both the 5x
    // logical-grid maps and 15-degree coverage around the complete vehicle.
    static void CreateVehicleSignedArcadeTimeFineDenseScene(string sourceScene, string destinationScene, string purpose)
    {
        const int refinement = 5;
        if (File.Exists(destinationScene))
        {
            Debug.LogWarning("Refined time/dense-lidar " + purpose +
                             " scene already exists; refusing to overwrite it: " + destinationScene);
            return;
        }
        if (!File.Exists(sourceScene))
        {
            Debug.LogError("Create the signed-pedal remaining-time " + purpose + " scene first: " + sourceScene);
            return;
        }
        if (!EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo()) return;
        if (!AssetDatabase.CopyAsset(sourceScene, destinationScene))
        {
            Debug.LogError("Could not copy " + sourceScene);
            return;
        }

        var scene = EditorSceneManager.OpenScene(destinationScene, OpenSceneMode.Single);
        var manager = Object.FindFirstObjectByType<TrafficScenarioManager>();
        GridManager grid = manager != null ? manager.gridManager : Object.FindFirstObjectByType<GridManager>();
        if (manager == null || grid == null || manager.cars == null || manager.cars.Length != 10)
        {
            Debug.LogError("Expected a ten-car TrafficScenarioManager and GridManager in " + destinationScene);
            return;
        }

        int sourceGridSize = grid.gridSize;
        float worldSizeM = grid.gridSize * grid.cellSize;
        grid.gridSize = sourceGridSize * refinement;
        grid.cellSize = worldSizeM / grid.gridSize;
        grid.voronoiMapRefinement = refinement;
        grid.smoothRefinedVoronoiBoundaries = true;
        grid.useCombinedMeshForRefinedGrid = true;
        grid.renderRefinedMeshInBatchMode = false;

        foreach (CarAgent car in manager.cars)
        {
            if (car == null || car.laserTileSensor == null || !car.includeRemainingStepsObservation)
            {
                Debug.LogError("Every traffic car needs LaserTileSensor and remaining-time observation.");
                return;
            }
            car.laserTileSensor.denseFullCircle15DegreeRays = true;
            var behavior = car.GetComponent<BehaviorParameters>();
            if (behavior == null)
            {
                Debug.LogError("Every traffic car needs BehaviorParameters.");
                return;
            }
            var serialized = new SerializedObject(behavior);
            serialized.FindProperty("m_BrainParameters")
                .FindPropertyRelative("VectorObservationSize").intValue =
                    17 + car.laserTileSensor.ActiveObservationCount + 1;
            serialized.ApplyModifiedPropertiesWithoutUndo();
        }

        EditorSceneManager.MarkSceneDirty(scene);
        EditorSceneManager.SaveScene(scene);
        Debug.Log("Created " + destinationScene + ": 330x330 refined logical map, 24 rays at 15 degrees, " +
                  "vehicle distances, and remaining time = 162 observations.");
    }

    // The original 66x66 source maps are deliberately not modified. The new
    // scenes load those exact assets into a 330x330 logical grid, preserving
    // the ~99m world size while making each terrain decision 0.3m wide.
    static void CreateVehicleSignedArcadeFineScene(string sourceScene, string destinationScene, string purpose)
    {
        const int refinement = 5;
        if (File.Exists(destinationScene))
        {
            Debug.LogWarning("5x refined signed-pedal " + purpose +
                             " scene already exists; refusing to overwrite it: " + destinationScene);
            return;
        }
        if (!File.Exists(sourceScene))
        {
            Debug.LogError("Create the signed-pedal arcade " + purpose + " scene first: " + sourceScene);
            return;
        }
        if (!EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo()) return;
        if (!AssetDatabase.CopyAsset(sourceScene, destinationScene))
        {
            Debug.LogError("Could not copy " + sourceScene);
            return;
        }

        var scene = EditorSceneManager.OpenScene(destinationScene, OpenSceneMode.Single);
        var manager = Object.FindFirstObjectByType<TrafficScenarioManager>();
        GridManager grid = manager != null ? manager.gridManager : Object.FindFirstObjectByType<GridManager>();
        if (manager == null || grid == null || manager.cars == null || manager.cars.Length != 10)
        {
            Debug.LogError("Expected a ten-car TrafficScenarioManager and GridManager in " + destinationScene);
            return;
        }

        int sourceGridSize = grid.gridSize;
        float worldSizeM = grid.gridSize * grid.cellSize;
        grid.gridSize = sourceGridSize * refinement;
        grid.cellSize = worldSizeM / grid.gridSize;
        grid.voronoiMapRefinement = refinement;
        grid.smoothRefinedVoronoiBoundaries = true;
        grid.useCombinedMeshForRefinedGrid = true;
        grid.renderRefinedMeshInBatchMode = false;

        EditorSceneManager.MarkSceneDirty(scene);
        EditorSceneManager.SaveScene(scene);
        Debug.Log("Created " + destinationScene + ": original " + sourceGridSize + "x" + sourceGridSize +
                  " text maps -> " + grid.gridSize + "x" + grid.gridSize + " logical grid at " +
                  grid.cellSize + "m cells; world remains " + worldSizeM + "m square.");
    }

    // This is intentionally a separate scene, rather than a runtime toggle:
    // adding the remaining-time value changes the policy input from 89 to 90.
    static void CreateVehicleSignedArcadeTimeScene(string sourceScene, string destinationScene, string purpose)
    {
        if (File.Exists(destinationScene))
        {
            Debug.LogWarning("Signed-pedal remaining-time " + purpose +
                             " scene already exists; refusing to overwrite it: " + destinationScene);
            return;
        }
        if (!File.Exists(sourceScene))
        {
            Debug.LogError("Create the signed-pedal arcade " + purpose + " scene first: " + sourceScene);
            return;
        }
        if (!EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo()) return;
        if (!AssetDatabase.CopyAsset(sourceScene, destinationScene))
        {
            Debug.LogError("Could not copy " + sourceScene);
            return;
        }

        var scene = EditorSceneManager.OpenScene(destinationScene, OpenSceneMode.Single);
        var manager = Object.FindFirstObjectByType<TrafficScenarioManager>();
        if (manager == null || manager.cars == null || manager.cars.Length != 10)
        {
            Debug.LogError("Expected ten traffic cars in " + destinationScene);
            return;
        }
        foreach (CarAgent car in manager.cars)
        {
            if (car == null || car.laserTileSensor == null)
            {
                Debug.LogError("Every traffic car needs CarAgent and LaserTileSensor.");
                return;
            }
            car.includeRemainingStepsObservation = true;
            var behavior = car.GetComponent<BehaviorParameters>();
            if (behavior == null)
            {
                Debug.LogError("Every traffic car needs BehaviorParameters.");
                return;
            }
            var serialized = new SerializedObject(behavior);
            serialized.FindProperty("m_BrainParameters")
                .FindPropertyRelative("VectorObservationSize").intValue =
                    17 + car.laserTileSensor.ActiveObservationCount + 1;
            serialized.ApplyModifiedPropertiesWithoutUndo();
        }
        EditorSceneManager.MarkSceneDirty(scene);
        EditorSceneManager.SaveScene(scene);
        Debug.Log("Created " + destinationScene +
                  " with 90 observations: the 89 vehicle-distance values plus shared episode time remaining.");
    }

    static void CreateVehicleSignedArcadeScene(string sourceScene, string destinationScene, string purpose)
    {
        if (File.Exists(destinationScene))
        {
            Debug.LogWarning("Signed-pedal arcade " + purpose + " scene already exists; refusing to overwrite it: " + destinationScene);
            return;
        }
        if (!File.Exists(sourceScene))
        {
            Debug.LogError("Create the vehicle-distance " + purpose + " scene first: " + sourceScene);
            return;
        }
        if (!EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo()) return;
        if (!AssetDatabase.CopyAsset(sourceScene, destinationScene))
        {
            Debug.LogError("Could not copy " + sourceScene);
            return;
        }
        var scene = EditorSceneManager.OpenScene(destinationScene, OpenSceneMode.Single);
        var manager = Object.FindFirstObjectByType<TrafficScenarioManager>();
        if (manager == null || manager.cars == null || manager.cars.Length != 10)
        {
            Debug.LogError("Expected ten traffic cars in " + destinationScene);
            return;
        }
        foreach (CarAgent car in manager.cars)
        {
            if (car == null || car.carController == null)
            {
                Debug.LogError("Every traffic car needs CarAgent and car_component.");
                return;
            }
            car.carController.arcadeAutomaticReverse = false;
            car.carController.signedPedalArcadeControls = true;
            var behavior = car.GetComponent<BehaviorParameters>();
            if (behavior == null)
            {
                Debug.LogError("Every traffic car needs BehaviorParameters.");
                return;
            }
            var serialized = new SerializedObject(behavior);
            var branches = serialized.FindProperty("m_BrainParameters")
                .FindPropertyRelative("m_ActionSpec")
                .FindPropertyRelative("BranchSizes");
            branches.arraySize = 0; // steering + signed pedal are both continuous.
            serialized.ApplyModifiedPropertiesWithoutUndo();
        }
        EditorSceneManager.MarkSceneDirty(scene);
        EditorSceneManager.SaveScene(scene);
        Debug.Log("Created " + destinationScene +
                  " with 89 observations, two continuous actions, and no discrete action branches.");
    }

    static void CreateVehicleArcadeScene(string sourceScene, string destinationScene, string purpose)
    {
        if (File.Exists(destinationScene))
        {
            Debug.LogWarning("Arcade " + purpose + " scene already exists; refusing to overwrite it: " +
                             destinationScene);
            return;
        }
        if (!File.Exists(sourceScene))
        {
            Debug.LogError("Create the vehicle-distance " + purpose + " scene first: " + sourceScene);
            return;
        }
        if (!EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo()) return;
        if (!AssetDatabase.CopyAsset(sourceScene, destinationScene))
        {
            Debug.LogError("Could not copy " + sourceScene);
            return;
        }

        var scene = EditorSceneManager.OpenScene(destinationScene, OpenSceneMode.Single);
        var manager = Object.FindFirstObjectByType<TrafficScenarioManager>();
        if (manager == null || manager.cars == null || manager.cars.Length != 10)
        {
            Debug.LogError("Expected ten traffic cars in " + destinationScene);
            return;
        }
        foreach (CarAgent car in manager.cars)
        {
            if (car == null || car.carController == null)
            {
                Debug.LogError("Every traffic car needs CarAgent and car_component.");
                return;
            }
            car.carController.arcadeAutomaticReverse = true;
            var behavior = car.GetComponent<BehaviorParameters>();
            if (behavior == null)
            {
                Debug.LogError("Every traffic car needs BehaviorParameters.");
                return;
            }
            var serialized = new SerializedObject(behavior);
            var branches = serialized.FindProperty("m_BrainParameters")
                .FindPropertyRelative("m_ActionSpec")
                .FindPropertyRelative("BranchSizes");
            branches.arraySize = 1;
            branches.GetArrayElementAtIndex(0).intValue = 2; // drive or brake
            serialized.ApplyModifiedPropertiesWithoutUndo();
        }
        EditorSceneManager.MarkSceneDirty(scene);
        EditorSceneManager.SaveScene(scene);
        Debug.Log("Created " + destinationScene +
                  " with 89 observations and one discrete [drive/brake] action branch.");
    }

    static void CreateVehicleDistanceScene(string sourceScene, string destinationScene)
    {
        if (File.Exists(destinationScene))
        {
            Debug.LogWarning("Vehicle-distance scene already exists; refusing to overwrite it: " +
                             destinationScene);
            return;
        }
        if (!File.Exists(sourceScene))
        {
            Debug.LogError("Create the source traffic scene first: " + sourceScene);
            return;
        }
        if (!EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo()) return;
        if (!AssetDatabase.CopyAsset(sourceScene, destinationScene))
        {
            Debug.LogError("Could not copy " + sourceScene);
            return;
        }
        var scene = EditorSceneManager.OpenScene(destinationScene, OpenSceneMode.Single);
        var manager = Object.FindFirstObjectByType<TrafficScenarioManager>();
        if (manager == null || manager.cars == null || manager.cars.Length != 10)
        {
            Debug.LogError("Expected a ten-car traffic scene in " + sourceScene);
            return;
        }
        foreach (CarAgent car in manager.cars)
        {
            if (car == null || car.laserTileSensor == null ||
                car.tileObservationMode != CarAgent.TileObservationMode.ContinuousLasers)
            {
                Debug.LogError("Every traffic car must use ContinuousLasers with a LaserTileSensor.");
                return;
            }
            car.laserTileSensor.includeVehicleDistanceObservations = true;
            car.laserTileSensor.vehicleDistanceScaleM = 10f;
            var behavior = car.GetComponent<BehaviorParameters>();
            if (behavior == null)
            {
                Debug.LogError("Every traffic car needs BehaviorParameters.");
                return;
            }
            var serialized = new SerializedObject(behavior);
            serialized.FindProperty("m_BrainParameters")
                .FindPropertyRelative("VectorObservationSize").intValue =
                    17 + car.laserTileSensor.ActiveObservationCount +
                    (car.includeRemainingStepsObservation ? 1 : 0);
            serialized.ApplyModifiedPropertiesWithoutUndo();
        }
        EditorSceneManager.MarkSceneDirty(scene);
        EditorSceneManager.SaveScene(scene);
        Debug.Log("Created " + destinationScene + " with 89 observations and vehicle-distance " +
                  "laser channels. Use laser_vehicle_distance_observation_enabled=1 in training YAML.");
    }

    [MenuItem("Tools/Traffic/Create shared-policy training scene")]
    public static void CreateTrainingScene()
    {
        if (File.Exists(TrainingScene))
        {
            Debug.LogWarning("Training scene already exists; refusing to overwrite it: " + TrainingScene);
            return;
        }
        if (!File.Exists(FullValidationScene) || !File.Exists(TrainingManifest))
        {
            Debug.LogError("Create the full-validation scene and generate " + TrainingManifest + " first.");
            return;
        }
        if (!EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo()) return;
        AssetDatabase.Refresh();
        TextAsset manifest = AssetDatabase.LoadAssetAtPath<TextAsset>(TrainingManifest);
        var cases = manifest != null
            ? JsonUtility.FromJson<TrafficScenarioManager.ScenarioCollection>(manifest.text) : null;
        if (cases?.scenarios == null || cases.scenarios.Length != 400 ||
            System.Array.Exists(cases.scenarios, s => string.IsNullOrEmpty(s.map_file)))
        {
            Debug.LogError("Training manifest must list all 400 VoronoiTrain maps.");
            return;
        }
        if (!AssetDatabase.CopyAsset(FullValidationScene, TrainingScene))
        {
            Debug.LogError("Could not copy " + FullValidationScene);
            return;
        }
        var scene = EditorSceneManager.OpenScene(TrainingScene, OpenSceneMode.Single);
        var manager = Object.FindFirstObjectByType<TrafficScenarioManager>();
        if (manager == null || manager.cars == null || manager.cars.Length != 10)
        {
            Debug.LogError("Expected a ten-car TrafficScenarioManager in the copied scene.");
            return;
        }
        manager.scenarioJson = manifest;
        manager.useValidationMaps = false;
        manager.randomizeScenarioEachEpisode = true;
        manager.useEnvironmentScenarioIndex = false;
        manager.randomizeRoutes = true;
        manager.defaultActiveCarCount = 2;
        foreach (CarAgent car in manager.cars) SetBehaviorName(car, "CarAgent");
        EditorSceneManager.MarkSceneDirty(scene);
        EditorSceneManager.SaveScene(scene);
        Debug.Log("Created " + TrainingScene + " with a shared CarAgent policy and training-only maps.");
    }

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
        }
        car0.gameObject.name = "TrafficCar0";
        SetBehaviorName(car0, "TrafficCar0");
        SetBehaviorName(car1, "TrafficCar1");
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
        manager.defaultActiveCarCount = 2;
        manager.editorDesiredPoolSize = 4;
        car0.trafficManager = manager;
        car0.trafficSeatIndex = 0;
        car1.trafficManager = manager;
        car1.trafficSeatIndex = 1;

        EditorSceneManager.MarkSceneDirty(scene);
        EditorSceneManager.SaveScene(scene);
        Debug.Log("Created " + TrafficScene + ". Build this scene for Python evaluation.");
    }

    [MenuItem("Tools/Traffic/Create all-validation traffic scene")]
    public static void CreateFullValidationScene()
    {
        if (File.Exists(FullValidationScene))
        {
            Debug.LogWarning("Full-validation scene already exists; refusing to overwrite it: " +
                             FullValidationScene);
            return;
        }
        if (!File.Exists(TrafficScene) || !File.Exists(FullValidationManifest))
        {
            Debug.LogError("Create the two-car traffic scene and generate " +
                           FullValidationManifest + " first.");
            return;
        }
        if (!EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo()) return;
        AssetDatabase.Refresh();
        TextAsset manifest = AssetDatabase.LoadAssetAtPath<TextAsset>(FullValidationManifest);
        if (manifest == null)
        {
            Debug.LogError("Could not load " + FullValidationManifest);
            return;
        }
        var cases = JsonUtility.FromJson<TrafficScenarioManager.ScenarioCollection>(manifest.text);
        if (cases?.scenarios == null || cases.scenarios.Length != 100 ||
            System.Array.Exists(cases.scenarios, s => string.IsNullOrEmpty(s.map_file)))
        {
            Debug.LogError("The full-validation manifest must list all 100 validation maps.");
            return;
        }
        if (!AssetDatabase.CopyAsset(TrafficScene, FullValidationScene))
        {
            Debug.LogError("Could not copy " + TrafficScene);
            return;
        }
        var scene = EditorSceneManager.OpenScene(FullValidationScene, OpenSceneMode.Single);
        var manager = Object.FindFirstObjectByType<TrafficScenarioManager>();
        if (manager == null)
        {
            Debug.LogError("Copied scene has no TrafficScenarioManager.");
            return;
        }
        manager.editorDesiredPoolSize = 10;
        ExpandCarPool(manager, manager.editorDesiredPoolSize);
        manager.scenarioJson = manifest;
        manager.randomizeRoutes = true;
        manager.minEndpointSeparationM = 8f;
        manager.minOwnSpawnGoalDistanceM = 20f;
        manager.placementAttempts = 50;
        manager.defaultActiveCarCount = 2;
        manager.scenarioIndex = 0;
        EditorSceneManager.MarkSceneDirty(scene);
        EditorSceneManager.SaveScene(scene);
        Debug.Log("Created " + FullValidationScene +
                  " with a ten-car pool and randomized Asphalt routes on all validation maps.");
    }

    [MenuItem("Tools/Traffic/Expand open traffic scene to desired car pool")]
    public static void ExpandOpenTrafficScene()
    {
        var scene = UnityEngine.SceneManagement.SceneManager.GetActiveScene();
        var manager = Object.FindFirstObjectByType<TrafficScenarioManager>();
        if (manager == null || manager.gameObject.scene != scene)
        {
            Debug.LogError("Open a traffic scene and set its manager's editorDesiredPoolSize first.");
            return;
        }
        if (manager.editorDesiredPoolSize < manager.cars.Length)
        {
            Debug.LogError("This tool only expands the pool. A smaller active count can be selected " +
                           "without removing cars, using -traffic-car-count.");
            return;
        }
        if (!EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo()) return;
        ExpandCarPool(manager, manager.editorDesiredPoolSize);
        EditorSceneManager.MarkSceneDirty(scene);
        EditorSceneManager.SaveScene(scene);
        Debug.Log($"Traffic car pool is now {manager.cars.Length}.");
    }

    static void ExpandCarPool(TrafficScenarioManager manager, int desiredCount)
    {
        if (manager.cars == null || manager.cars.Length < 2 || desiredCount < manager.cars.Length)
            throw new System.InvalidOperationException("Expected a two-or-more-car traffic scene.");
        var pool = new List<CarAgent>(manager.cars);
        CarAgent template = pool[0];
        for (int seat = pool.Count; seat < desiredCount; seat++)
        {
            GameObject duplicate = Object.Instantiate(template.gameObject);
            duplicate.name = $"TrafficCar{seat}";
            CarAgent car = duplicate.GetComponent<CarAgent>();
            GameObject goal = Object.Instantiate(template.goal.gameObject);
            goal.name = $"TrafficGoal{seat}";
            car.goal = goal.transform;
            GameObject spawn = Object.Instantiate(template.spawnPoint.gameObject);
            spawn.name = $"TrafficSpawn{seat}";
            car.spawnPoint = spawn.transform;
            car.trafficManager = manager;
            car.trafficSeatIndex = seat;
            car.MaxStep = 0;
            SetBehaviorName(car, $"TrafficCar{seat}");
            foreach (var component in duplicate.GetComponents<ManualDrivingView>()) component.enabled = false;
            foreach (var component in duplicate.GetComponents<SensorWeightOverlay>()) component.enabled = false;
            foreach (var component in duplicate.GetComponents<BrakeDistanceTest>()) component.enabled = false;
            duplicate.SetActive(false); // The manager enables selected seats at startup.
            goal.SetActive(false);
            spawn.SetActive(false);
            pool.Add(car);
        }
        manager.cars = pool.ToArray();
    }

    static void SetBehaviorName(CarAgent car, string name)
    {
        var serialized = new SerializedObject(car.GetComponent<BehaviorParameters>());
        serialized.FindProperty("m_BehaviorName").stringValue = name;
        serialized.ApplyModifiedPropertiesWithoutUndo();
    }
}
