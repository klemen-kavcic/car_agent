using System.Collections.Generic;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

// Copies the material overrides made on TrafficCar0's imported Tesla hierarchy
// to every other pooled traffic car. This intentionally touches Renderer
// materials only: it cannot change colliders, ML-Agents settings, or physics.
public static class TrafficCarMaterialCopyTool
{
    [MenuItem("Tools/Traffic/Copy TrafficCar0 materials to all traffic cars")]
    public static void CopyTrafficCar0Materials()
    {
        TrafficScenarioManager manager = Object.FindFirstObjectByType<TrafficScenarioManager>();
        if (manager == null || manager.cars == null || manager.cars.Length < 2 || manager.cars[0] == null)
        {
            Debug.LogError("Open a traffic scene with TrafficScenarioManager and TrafficCar0 first.");
            return;
        }

        Transform sourceRoot = manager.cars[0].transform;
        var sourceMaterials = new Dictionary<string, Material[]>();
        foreach (Renderer renderer in sourceRoot.GetComponentsInChildren<Renderer>(true))
        {
            string path = AnimationUtility.CalculateTransformPath(renderer.transform, sourceRoot);
            sourceMaterials[path] = renderer.sharedMaterials;
        }

        int copiedRenderers = 0;
        int missingPaths = 0;
        for (int carIndex = 1; carIndex < manager.cars.Length; carIndex++)
        {
            CarAgent car = manager.cars[carIndex];
            if (car == null) continue;

            foreach (Renderer renderer in car.GetComponentsInChildren<Renderer>(true))
            {
                string path = AnimationUtility.CalculateTransformPath(renderer.transform, car.transform);
                if (!sourceMaterials.TryGetValue(path, out Material[] materials))
                {
                    missingPaths++;
                    continue;
                }

                Undo.RecordObject(renderer, "Copy TrafficCar0 materials");
                renderer.sharedMaterials = materials;
                EditorUtility.SetDirty(renderer);
                copiedRenderers++;
            }
        }

        EditorSceneManager.MarkSceneDirty(SceneManager.GetActiveScene());
        Debug.Log($"Copied TrafficCar0 material assignments to {manager.cars.Length - 1} cars " +
            $"({copiedRenderers} renderers; {missingPaths} unmatched hierarchy paths).");
    }
}
