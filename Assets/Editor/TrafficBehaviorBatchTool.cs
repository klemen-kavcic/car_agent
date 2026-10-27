using System.Collections.Generic;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;
using Unity.MLAgents.Policies;
using Unity.InferenceEngine;

// A scene-authoring convenience tool. Apply an ONNX model and inference
// settings to all pooled traffic cars in one operation before entering Play
// mode, rather than editing ten BehaviorParameters components by hand.
public sealed class TrafficBehaviorBatchTool : EditorWindow
{
    ModelAsset model;
    BehaviorType behaviorType = BehaviorType.InferenceOnly;
    bool deterministicInference = true;

    [MenuItem("Tools/Traffic/Configure all traffic-car behaviours...")]
    static void Open()
    {
        GetWindow<TrafficBehaviorBatchTool>("Traffic car behaviour");
    }

    void OnGUI()
    {
        var manager = Object.FindFirstObjectByType<TrafficScenarioManager>();
        EditorGUILayout.LabelField("Traffic-car batch configuration", EditorStyles.boldLabel);
        EditorGUILayout.Space(3);

        if (EditorApplication.isPlaying)
        {
            EditorGUILayout.HelpBox(
                "Leave Play mode before applying settings. This tool saves the settings to the scene so every car starts with the same policy.",
                MessageType.Info);
            return;
        }
        if (manager == null || manager.cars == null || manager.cars.Length == 0)
        {
            EditorGUILayout.HelpBox("Open a traffic scene containing TrafficScenarioManager first.", MessageType.Error);
            return;
        }

        model = (ModelAsset)EditorGUILayout.ObjectField("ONNX model", model, typeof(ModelAsset), false);
        behaviorType = (BehaviorType)EditorGUILayout.EnumPopup("Behaviour type", behaviorType);
        deterministicInference = EditorGUILayout.Toggle("Deterministic inference", deterministicInference);

        EditorGUILayout.Space(5);
        int activeCount = CountActiveCars(manager);
        EditorGUILayout.HelpBox(
            $"Pool: {manager.cars.Length} cars. Currently active in this saved scene: {activeCount}.\n" +
            "Use Inference Only for watching an ONNX policy. Deterministic picks its mean/greedy action; stochastic samples actions.",
            MessageType.None);

        bool needModel = behaviorType == BehaviorType.InferenceOnly && model == null;
        using (new EditorGUI.DisabledScope(needModel))
        {
            if (GUILayout.Button($"Apply to active cars ({activeCount})"))
                Apply(manager, activeOnly: true);
            if (GUILayout.Button($"Apply to all pooled cars ({manager.cars.Length})"))
                Apply(manager, activeOnly: false);
        }
        if (needModel)
            EditorGUILayout.HelpBox("Choose an ONNX model before using Inference Only.", MessageType.Warning);

        EditorGUILayout.Space(5);
        EditorGUILayout.HelpBox(
            "Choose a model whose observation size and action layout match this scene. For example, the signed-pedal model belongs in the signed-arcade scene (two continuous actions, no discrete branch).",
            MessageType.Info);
    }

    static int CountActiveCars(TrafficScenarioManager manager)
    {
        int count = 0;
        foreach (CarAgent car in manager.cars)
            if (car != null && car.gameObject.activeSelf) count++;
        return count;
    }

    void Apply(TrafficScenarioManager manager, bool activeOnly)
    {
        var changed = new List<BehaviorParameters>();
        foreach (CarAgent car in manager.cars)
        {
            if (car == null || (activeOnly && !car.gameObject.activeSelf)) continue;
            BehaviorParameters behavior = car.GetComponent<BehaviorParameters>();
            if (behavior == null)
            {
                Debug.LogWarning($"{car.name} has no BehaviorParameters; skipped.", car);
                continue;
            }

            Undo.RecordObject(behavior, "Configure traffic-car behaviour");
            behavior.Model = model;
            behavior.BehaviorType = behaviorType;
            behavior.DeterministicInference = deterministicInference;
            EditorUtility.SetDirty(behavior);
            changed.Add(behavior);
        }

        if (changed.Count == 0)
        {
            Debug.LogWarning("No traffic-car behaviour settings were changed.");
            return;
        }

        EditorSceneManager.MarkSceneDirty(SceneManager.GetActiveScene());
        Debug.Log($"Applied {(model == null ? "no model" : model.name)} / {behaviorType} / " +
                  $"deterministic={deterministicInference} to {changed.Count} traffic cars.");
    }
}
