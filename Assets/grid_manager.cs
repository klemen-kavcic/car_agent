using UnityEngine;
using System.Collections.Generic;
using UnityEngine.Rendering;

public enum MapSource { Perlin, Voronoi }

public class GridManager : MonoBehaviour
{
    public int gridSize = 20;
    public float cellSize = 5f;
    public GameObject tilePrefab;

    [Header("Refined Voronoi map mode")]
    [Tooltip("Integer factor used only when loading the fixed Voronoi .txt maps. A value of 5 lets a 66x66 source map drive a 330x330 logical grid while retaining the same world dimensions. Leave at 1 for the existing behaviour.")]
    [Min(1)] public int voronoiMapRefinement = 1;

    [Tooltip("At refinement > 1, interpolate categorical source-map corners before choosing each fine logical cell. This rounds diagonal/corner boundaries instead of merely repeating square source cells.")]
    public bool smoothRefinedVoronoiBoundaries = true;

    [Tooltip("Render refined grids as five combined meshes instead of one GameObject per cell. Logical tile queries, sensors, rewards, and terminal checks still use every fine cell.")]
    public bool useCombinedMeshForRefinedGrid = true;

    [Tooltip("Normally refined tile visuals are omitted in headless/batch players because they do not affect training. Enable only when a batch build genuinely needs rendered terrain.")]
    public bool renderRefinedMeshInBatchMode = false;

    [Tooltip("Perlin = existing procedural noise terrain. Voronoi = prebaked road-network maps " +
        "loaded at random from Resources/Maps/Voronoi (see Map Gen/voronoi_curved.py's --gridsize " +
        "export, which must match gridSize below).")]
    public MapSource mapSource = MapSource.Perlin;

    // The "safe default" tile type, used both as what forceAllNormal blanks the whole grid to
    // (despite the name - see GenerateGrid) and as the type CarAgent's spawn/goal placement
    // retries against (see IsSpawnableTile in carAgent.cs). Always Asphalt now that Normal has
    // been merged into it - both were physics-identical (no special case in car_component.cs)
    // and played this exact same role, just under different mapSources.
    public TileType SpawnableTileType => TileType.Asphalt;

    [Header("Tile Materials")]
    public Material slipperyMaterial;
    public Material speedLimitedMaterial;
    public Material terminalMaterial;
    public Material asphaltMaterial;
    public Material gravelMaterial;

    // Field name kept as probNormal (not renamed to probAsphalt) so the value already set in the
    // Inspector isn't silently reset to default - Unity matches serialized fields by name.
    [Header("Tile Distribution (auto-normalized to sum to 1, Perlin source only - probNormal is Asphalt's weight)")]
    [Range(0f, 1f)] public float probNormal       = 0.40f;
    [Range(0f, 1f)] public float probSpeedLimited  = 0.20f;
    [Range(0f, 1f)] public float probSlippery      = 0.20f;
    [Range(0f, 1f)] public float probTerminal      = 0.20f;

    [Header("Perlin Noise")]
    [Tooltip("Controls region size. Low (~0.1) = large blobs, high (~0.5) = fragmented patches. Lower this if you increase grid resolution.")]
    [Range(0.02f, 0.8f)] public float noiseScale = 0.15f;

    [Tooltip("When true (set by CarAgent during its bootstrap curriculum), every generated tile is " +
        "forced to SpawnableTileType (Asphalt) regardless of probXxx settings. GridManager has no " +
        "ML-Agents dependency and doesn't know why - it just does what it's told.")]
    public bool forceAllNormal = false;

    [HideInInspector] public TileType[,] tileTypes;
    private GridTile[,] tiles;
    private Renderer[] refinedMeshRenderers;
    // Reused while turning a categorical source map into fine cells; allocating
    // a new small array for each of 108,900 cells would create avoidable GC work.
    private readonly float[] refinedSampleScores = new float[5];
    private Vector3 gridOrigin;
    // Rendering can be suppressed for the manual observation-only view without changing the
    // colliders, tileTypes array, or any physics/reward logic.
    private bool tileVisualsVisible = true;

    // Two independent noise offsets: one for passability, one for terrain type
    private float passOffsetX, passOffsetZ;
    private float terrainOffsetX, terrainOffsetZ;

    [Tooltip("When true (set by CarAgent from the use_validation_maps environment parameter), " +
        "Voronoi maps are loaded from Resources/Maps/VoronoiVal instead of Maps/VoronoiTrain - a " +
        "held-out set never seen during training, for eval_checkpoints.py to measure " +
        "generalisation against instead of memorisation of the training pool. GridManager has no " +
        "ML-Agents dependency and doesn't know why - same pattern as forceAllNormal above.")]
    public bool useValidationMaps = false;

    [Tooltip("Use the small, hand-authored Maps/TrafficRecovery lessons. This is set only by " +
        "TrafficScenarioManager's optional recovery curriculum and is separate from both normal " +
        "training and held-out validation maps.")]
    public bool useRecoveryMaps = false;

    [Tooltip("When >= 0 (set by CarAgent from the fixed_map_index environment parameter, only " +
        "while fixed_eval_enabled is set), forces BuildVoronoiPlan() to load this exact index " +
        "from the active mapAssets array (train or val, per useValidationMaps) instead of picking " +
        "randomly - used by eval_trajectories.py so the same fixed map is reloaded on every " +
        "episode/rollout. -1 (default) preserves the existing random behavior. Falls back to " +
        "random (with a warning) if out of range for the currently-loaded array. Same " +
        "GridManager-doesn't-know-why pattern as useValidationMaps above.")]
    public int forcedMapIndex = -1;
    // Actual successfully parsed Voronoi asset index, not merely the requested index.
    // -1 means no asset was loaded or parsing fell back to the all-Asphalt plan.
    public int LastLoadedVoronoiMapIndex { get; private set; } = -1;
    public string LastLoadedVoronoiMapName { get; private set; } = "";

    // Cached separately on first use per set, so every Regenerate() doesn't re-hit
    // Resources.LoadAll, and flipping useValidationMaps mid-run doesn't force a reload of the set
    // that's already cached.
    private TextAsset[] voronoiTrainMapAssets;
    private TextAsset[] voronoiValMapAssets;
    private TextAsset[] recoveryMapAssets;

    // Index = the integer code voronoi_curved.py's classify_grid()/export_unity_grid() writes
    // per cell: 0=gravel(BG), 1=asphalt(road), 2=grass(green blob), 3=ice(blue blob),
    // 4=terminal(red blob). Grass maps to SpeedLimited (not Normal) - same reduced-top-speed
    // behaviour and material (Mat_grass) as SpeedLimited already has under Perlin.
    private static readonly TileType[] VoronoiCodeToType =
    {
        TileType.Gravel,
        TileType.Asphalt,
        TileType.SpeedLimited,
        TileType.Slippery,
        TileType.Terminal,
    };

    void Awake()
    {
        gridOrigin = new Vector3(-(gridSize * cellSize) / 2f, 0f, -(gridSize * cellSize) / 2f);
        tileTypes = new TileType[gridSize, gridSize];
        tiles = null;
        RandomiseNoiseOffset();
        GenerateGrid();
    }

    public void Regenerate()
    {
        LastLoadedVoronoiMapIndex = -1;
        LastLoadedVoronoiMapName = "";
        gridOrigin = new Vector3(-(gridSize * cellSize) / 2f, 0f, -(gridSize * cellSize) / 2f);
        foreach (Transform child in transform)
            Destroy(child.gameObject);
        tileTypes = new TileType[gridSize, gridSize];
        tiles = null;
        refinedMeshRenderers = null;
        RandomiseNoiseOffset();
        GenerateGrid();
    }

    void RandomiseNoiseOffset()
    {
        passOffsetX   = Random.Range(0f, 9999f);
        passOffsetZ   = Random.Range(0f, 9999f);
        terrainOffsetX = Random.Range(0f, 9999f);
        terrainOffsetZ = Random.Range(0f, 9999f);
    }

    void GenerateGrid()
    {
        TileType[,] plan = mapSource == MapSource.Voronoi ? BuildVoronoiPlan() : BuildPerlinPlan();

        bool combinedRefinedMesh = mapSource == MapSource.Voronoi &&
                                   voronoiMapRefinement > 1 &&
                                   useCombinedMeshForRefinedGrid;

        // The high-resolution data remains fully real even when its visuals are
        // suppressed in a headless training player. This avoids spawning 108,900
        // visual GameObjects for a 330x330 grid in every parallel environment.
        if (combinedRefinedMesh)
        {
            for (int x = 0; x < gridSize; x++)
                for (int z = 0; z < gridSize; z++)
                    tileTypes[x, z] = forceAllNormal ? SpawnableTileType : plan[x, z];

            if (!Application.isBatchMode || renderRefinedMeshInBatchMode)
                BuildCombinedRefinedMeshes();
            return;
        }

        tiles = new GridTile[gridSize, gridSize];

        for (int x = 0; x < gridSize; x++)
        {
            for (int z = 0; z < gridSize; z++)
            {
                TileType type = forceAllNormal ? SpawnableTileType : plan[x, z];

                Vector3 pos = gridOrigin + new Vector3(
                    x * cellSize + cellSize * 0.5f,
                    0f,
                    z * cellSize + cellSize * 0.5f
                );

                var go = Instantiate(tilePrefab, pos, Quaternion.identity, transform);
                go.transform.localScale = new Vector3(cellSize / 10f, 1f, cellSize / 10f);

                var tile = go.AddComponent<GridTile>();
                tile.coord = new Vector2Int(x, z);
                tile.type = type;

                var mat = MaterialForType(type);
                Renderer renderer = go.GetComponent<Renderer>();
                if (mat != null && renderer != null)
                    renderer.material = mat;
                if (renderer != null)
                    renderer.enabled = tileVisualsVisible;

                tileTypes[x, z] = type;
                tiles[x, z] = tile;
            }
        }
    }

    TileType[,] BuildPerlinPlan()
    {
        var plan = new TileType[gridSize, gridSize];

        float total = probNormal + probSpeedLimited + probSlippery + probTerminal;
        if (total <= 0f) total = 1f;

        // Noise 1 threshold: above this → Terminal (high noise = impassable blobs)
        float terminalThreshold = 1f - (probTerminal / total);

        // Noise 2 ordering: SpeedLimited → Asphalt → Slippery
        // Road sits in the middle so ice can border it directly on one side, grass on the other.
        float terrainTotal = probNormal + probSpeedLimited + probSlippery;
        if (terrainTotal <= 0f) terrainTotal = 1f;
        float tN  = probSpeedLimited / terrainTotal;          // below = SpeedLimited
        float tNS = tN + probNormal / terrainTotal;           // below = Asphalt, above = Slippery

        for (int x = 0; x < gridSize; x++)
        {
            for (int z = 0; z < gridSize; z++)
            {
                float passNoise = Mathf.PerlinNoise(
                    (x + passOffsetX) * noiseScale,
                    (z + passOffsetZ) * noiseScale
                );

                if (passNoise >= terminalThreshold)
                {
                    plan[x, z] = TileType.Terminal;
                }
                else
                {
                    float terrainNoise = Mathf.PerlinNoise(
                        (x + terrainOffsetX) * noiseScale,
                        (z + terrainOffsetZ) * noiseScale
                    );

                    if      (terrainNoise < tN)  plan[x, z] = TileType.SpeedLimited;
                    else if (terrainNoise < tNS) plan[x, z] = TileType.Asphalt;
                    else                         plan[x, z] = TileType.Slippery;
                }
            }
        }
        return plan;
    }

    // Resources.LoadAll's return order isn't formally guaranteed by Unity across platforms/builds.
    // Sorting by asset name here makes forcedMapIndex mean something stable and predictable
    // (matching whatever sorted-filename order eval_trajectories.py's candidate generator used),
    // instead of relying on LoadAll's incidental order happening to match.
    static TextAsset[] SortedMapAssets(string resourceFolder, string requiredNamePrefix = null)
    {
        var assets = Resources.LoadAll<TextAsset>(resourceFolder);
        if (!string.IsNullOrEmpty(requiredNamePrefix))
            assets = System.Array.FindAll(assets, asset =>
                asset.name.StartsWith(requiredNamePrefix) &&
                asset.name.Length > requiredNamePrefix.Length &&
                char.IsDigit(asset.name[requiredNamePrefix.Length]));
        System.Array.Sort(assets, (a, b) => string.CompareOrdinal(a.name, b.name));
        return assets;
    }

    // Recovery contains a human-readable JSON catalogue next to the .txt maps.
    // Filter by the map filename prefix so that metadata can never become a
    // selectable grid or shift an otherwise stable forced-map index.
    public int RecoveryMapIndex(string mapName)
    {
        if (recoveryMapAssets == null)
            recoveryMapAssets = SortedMapAssets("Maps/TrafficRecovery", "recovery_");
        string bareName = System.IO.Path.GetFileNameWithoutExtension(mapName);
        for (int i = 0; i < recoveryMapAssets.Length; i++)
            if (recoveryMapAssets[i].name == bareName) return i;
        return -1;
    }

    // Loads a random prebaked map exported by Map Gen/voronoi_curved.py or voronoi_highways.py
    // (see VoronoiCodeToType for the code→TileType mapping) from Resources/Maps/VoronoiTrain, or
    // Maps/VoronoiVal when useValidationMaps is set. Falls back to an all-Asphalt plan (logging a
    // warning) if no maps are bundled or a file doesn't parse cleanly against the current
    // gridSize - this deliberately never throws, since a bad map file shouldn't crash an HPC
    // training run mid-episode.
    TileType[,] BuildVoronoiPlan()
    {
        LastLoadedVoronoiMapIndex = -1;
        LastLoadedVoronoiMapName = "";
        var plan = new TileType[gridSize, gridSize];

        string resourceFolder;
        TextAsset[] mapAssets;
        if (useRecoveryMaps)
        {
            resourceFolder = "Maps/TrafficRecovery";
            if (recoveryMapAssets == null)
                recoveryMapAssets = SortedMapAssets(resourceFolder, "recovery_");
            mapAssets = recoveryMapAssets;
        }
        else if (useValidationMaps)
        {
            resourceFolder = "Maps/VoronoiVal";
            if (voronoiValMapAssets == null)
                voronoiValMapAssets = SortedMapAssets(resourceFolder);
            mapAssets = voronoiValMapAssets;
        }
        else
        {
            resourceFolder = "Maps/VoronoiTrain";
            if (voronoiTrainMapAssets == null)
                voronoiTrainMapAssets = SortedMapAssets(resourceFolder);
            mapAssets = voronoiTrainMapAssets;
        }

        if (mapAssets == null || mapAssets.Length == 0)
        {
            Debug.LogWarning($"[GridManager] mapSource=Voronoi but no maps found under " +
                $"Resources/{resourceFolder} - falling back to an all-Asphalt grid.");
            FillAllAsphalt(plan);
            return plan;
        }

        int mapIndex = Random.Range(0, mapAssets.Length);
        if (forcedMapIndex >= 0)
        {
            if (forcedMapIndex < mapAssets.Length)
            {
                mapIndex = forcedMapIndex;
            }
            else
            {
                Debug.LogWarning($"[GridManager] forcedMapIndex={forcedMapIndex} out of range for " +
                    $"{mapAssets.Length} maps in Resources/{resourceFolder} - falling back to random.");
            }
        }
        var asset = mapAssets[mapIndex];
        if (forcedMapIndex >= 0)
        {
            // eval_trajectories.py's candidate generator assigns map_index by the SAME sorted-by-
            // name order SortedMapAssets() below produces, so index N here should always resolve
            // to the same map that generator meant - logged every time so a human can cross-check
            // (see the plan's verification step) rather than trusting that silently.
            Debug.Log($"[GridManager] forcedMapIndex={forcedMapIndex} resolved to map '{asset.name}'.");
        }
        string[] rows = asset.text.Trim().Split('\n');
        int sourceSize = rows.Length;
        bool refined = voronoiMapRefinement > 1;
        int expectedGridSize = refined ? sourceSize * voronoiMapRefinement : sourceSize;
        if (sourceSize == 0 || gridSize != expectedGridSize)
        {
            Debug.LogWarning($"[GridManager] Voronoi map '{asset.name}' has {sourceSize} rows, " +
                $"but this scene needs gridSize={expectedGridSize} " +
                $"(source x refinement {voronoiMapRefinement}) and has {gridSize}; falling back to all Asphalt.");
            FillAllAsphalt(plan);
            return plan;
        }

        var sourcePlan = new TileType[sourceSize, sourceSize];
        for (int z = 0; z < sourceSize; z++)
        {
            string[] cols = rows[z].Trim().Split((char[])null, System.StringSplitOptions.RemoveEmptyEntries);
            if (cols.Length != sourceSize)
            {
                Debug.LogWarning($"[GridManager] Voronoi map '{asset.name}' row {z} has " +
                    $"{cols.Length} cells, expected source size {sourceSize} - falling back to an " +
                    "all-Asphalt grid.");
                FillAllAsphalt(plan);
                return plan;
            }

            for (int x = 0; x < sourceSize; x++)
            {
                if (!int.TryParse(cols[x], out int code) || code < 0 || code >= VoronoiCodeToType.Length)
                {
                    Debug.LogWarning($"[GridManager] Voronoi map '{asset.name}' has an invalid " +
                        $"cell code '{cols[x]}' at ({x},{z}) - treating as Gravel.");
                    code = 0;
                }
                sourcePlan[x, z] = VoronoiCodeToType[code];
            }
        }

        if (!refined)
        {
            plan = sourcePlan;
        }
        else
        {
            for (int x = 0; x < gridSize; x++)
                for (int z = 0; z < gridSize; z++)
                    plan[x, z] = SampleRefinedSourceMap(sourcePlan, x, z, voronoiMapRefinement);
        }
        LastLoadedVoronoiMapIndex = mapIndex;
        LastLoadedVoronoiMapName = asset.name;
        return plan;
    }

    // The original map remains the source of truth. Fine-cell centres are
    // sampled from its categorical field. Bilinear scores turn a coarse corner
    // transition into a diagonal/rounded high-resolution boundary rather than
    // a block of repeated 1.5m squares.
    TileType SampleRefinedSourceMap(TileType[,] source, int fineX, int fineZ, int refinement)
    {
        if (!smoothRefinedVoronoiBoundaries)
            return source[Mathf.Clamp(fineX / refinement, 0, source.GetLength(0) - 1),
                          Mathf.Clamp(fineZ / refinement, 0, source.GetLength(1) - 1)];

        float sx = (fineX + 0.5f) / refinement - 0.5f;
        float sz = (fineZ + 0.5f) / refinement - 0.5f;
        int x0 = Mathf.FloorToInt(sx), z0 = Mathf.FloorToInt(sz);
        float tx = sx - x0, tz = sz - z0;
        int x1 = x0 + 1, z1 = z0 + 1;
        x0 = Mathf.Clamp(x0, 0, source.GetLength(0) - 1);
        x1 = Mathf.Clamp(x1, 0, source.GetLength(0) - 1);
        z0 = Mathf.Clamp(z0, 0, source.GetLength(1) - 1);
        z1 = Mathf.Clamp(z1, 0, source.GetLength(1) - 1);

        System.Array.Clear(refinedSampleScores, 0, refinedSampleScores.Length);
        AddScore(refinedSampleScores, source[x0, z0], (1f - tx) * (1f - tz));
        AddScore(refinedSampleScores, source[x1, z0], tx * (1f - tz));
        AddScore(refinedSampleScores, source[x0, z1], (1f - tx) * tz);
        AddScore(refinedSampleScores, source[x1, z1], tx * tz);

        int winner = 0;
        for (int i = 1; i < refinedSampleScores.Length; i++)
            if (refinedSampleScores[i] > refinedSampleScores[winner]) winner = i;
        return (TileType)winner;
    }

    static void AddScore(float[] scores, TileType type, float weight)
    {
        int index = (int)type;
        if (index >= 0 && index < scores.Length) scores[index] += weight;
    }

    void BuildCombinedRefinedMeshes()
    {
        var renderers = new List<Renderer>();
        for (int typeIndex = 0; typeIndex < 5; typeIndex++)
        {
            TileType type = (TileType)typeIndex;
            var vertices = new List<Vector3>();
            var triangles = new List<int>();
            var uvs = new List<Vector2>();
            for (int x = 0; x < gridSize; x++)
            {
                for (int z = 0; z < gridSize; z++)
                {
                    if (tileTypes[x, z] != type) continue;
                    int start = vertices.Count;
                    float x0 = gridOrigin.x + x * cellSize;
                    float z0 = gridOrigin.z + z * cellSize;
                    vertices.Add(new Vector3(x0, 0f, z0));
                    vertices.Add(new Vector3(x0 + cellSize, 0f, z0));
                    vertices.Add(new Vector3(x0 + cellSize, 0f, z0 + cellSize));
                    vertices.Add(new Vector3(x0, 0f, z0 + cellSize));
                    uvs.Add(Vector2.zero); uvs.Add(Vector2.right); uvs.Add(Vector2.one); uvs.Add(Vector2.up);
                    triangles.Add(start); triangles.Add(start + 2); triangles.Add(start + 1);
                    triangles.Add(start); triangles.Add(start + 3); triangles.Add(start + 2);
                }
            }
            if (vertices.Count == 0) continue;
            var meshObject = new GameObject("Refined " + type + " tiles");
            meshObject.transform.SetParent(transform, false);
            var filter = meshObject.AddComponent<MeshFilter>();
            var renderer = meshObject.AddComponent<MeshRenderer>();
            var mesh = new Mesh { name = "Refined " + type + " grid" };
            mesh.indexFormat = IndexFormat.UInt32;
            mesh.SetVertices(vertices);
            mesh.SetUVs(0, uvs);
            mesh.SetTriangles(triangles, 0, true);
            mesh.RecalculateBounds();
            filter.sharedMesh = mesh;
            // The old per-tile prefab supplied the ground collision surface.
            // A combined visual mesh needs its own static collider so cars and
            // wheel colliders do not fall through the refined environment.
            var collider = meshObject.AddComponent<MeshCollider>();
            collider.sharedMesh = mesh;
            renderer.sharedMaterial = MaterialForType(type);
            renderer.enabled = tileVisualsVisible;
            renderers.Add(renderer);
        }
        refinedMeshRenderers = renderers.ToArray();
    }

    void FillAllAsphalt(TileType[,] plan)
    {
        for (int x = 0; x < gridSize; x++)
            for (int z = 0; z < gridSize; z++)
                plan[x, z] = TileType.Asphalt;
    }

    Material MaterialForType(TileType type)
    {
        switch (type)
        {
            case TileType.Slippery:     return slipperyMaterial;
            case TileType.SpeedLimited: return speedLimitedMaterial;
            case TileType.Terminal:     return terminalMaterial;
            case TileType.Asphalt:      return asphaltMaterial;
            case TileType.Gravel:       return gravelMaterial;
            default:                    return asphaltMaterial;
        }
    }

    // Unclamped: callers must bounds-check (see GetTileAt) so off-grid positions are distinguishable.
    public Vector2Int WorldToGrid(Vector3 worldPos)
    {
        int x = Mathf.FloorToInt((worldPos.x - gridOrigin.x) / cellSize);
        int z = Mathf.FloorToInt((worldPos.z - gridOrigin.z) / cellSize);
        return new Vector2Int(x, z);
    }

    // Read-only geometry helpers for continuous grid sensors. Keeping the origin private avoids
    // accidental edits, while letting a sensor traverse the exact same logical cells as GetTileAt.
    public Vector3 GridOrigin => gridOrigin;
    public float GridDiagonal => Mathf.Sqrt(2f) * gridSize * cellSize;

    public bool IsInBounds(Vector2Int cell)
    {
        return tileTypes != null && cell.x >= 0 && cell.x < tileTypes.GetLength(0) &&
               cell.y >= 0 && cell.y < tileTypes.GetLength(1);
    }

    public TileType GetTileAtCell(Vector2Int cell)
    {
        return IsInBounds(cell) ? tileTypes[cell.x, cell.y] : TileType.Terminal;
    }

    public TileType GetTileAt(Vector3 worldPos)
    {
        var grid = tileTypes;
        if (grid == null) return TileType.Asphalt;
        var c = WorldToGrid(worldPos);
        if (c.x < 0 || c.x >= grid.GetLength(0) || c.y < 0 || c.y >= grid.GetLength(1))
            return TileType.Terminal; // off the map — treated as a hazard so the agent can see/avoid the edge
        return grid[c.x, c.y];
    }

    // Used by ManualDrivingView. Rendering is deliberately separate from tile state: hiding the
    // map must not give the human a different physical world from the one the policy experiences.
    public void SetTileVisualsVisible(bool visible)
    {
        tileVisualsVisible = visible;
        if (refinedMeshRenderers != null)
            foreach (Renderer renderer in refinedMeshRenderers)
                if (renderer != null) renderer.enabled = visible;
        if (tiles == null) return;
        for (int x = 0; x < tiles.GetLength(0); x++)
        {
            for (int z = 0; z < tiles.GetLength(1); z++)
            {
                GridTile tile = tiles[x, z];
                if (tile == null) continue;
                Renderer renderer = tile.GetComponent<Renderer>();
                if (renderer != null) renderer.enabled = visible;
            }
        }
    }

    // Overrides a single already-generated tile's type (and material) in place, without a full
    // Regenerate() - used by the bootstrap curriculum to force one specific cell (e.g. directly in
    // front of a Behind-stage spawn) regardless of forceAllNormal/the probXxx tile mix. Silently
    // no-ops if worldPos falls outside the grid.
    public void SetTileTypeAt(Vector3 worldPos, TileType type)
    {
        var c = WorldToGrid(worldPos);
        if (c.x < 0 || c.x >= gridSize || c.y < 0 || c.y >= gridSize) return;

        tileTypes[c.x, c.y] = type;
        if (refinedMeshRenderers != null)
        {
            // Refined traffic scenes do not use dynamic bootstrap tiles. Keep
            // the logical world authoritative here; rebuilding 330x330 meshes
            // for a one-cell lesson edit would be needlessly expensive.
            return;
        }
        var tile = tiles[c.x, c.y];
        if (tile == null) return;
        tile.type = type;

        var mat = MaterialForType(type);
        if (mat != null)
        {
            var renderer = tile.GetComponent<Renderer>();
            if (renderer != null) renderer.material = mat;
        }
    }

    public void GetNeighborhood5x5(Vector3 worldPos, TileType[] result)
    {
        var center = WorldToGrid(worldPos);
        int i = 0;
        for (int dz = -2; dz <= 2; dz++)
        {
            for (int dx = -2; dx <= 2; dx++)
            {
                int nx = center.x + dx;
                int nz = center.y + dz;
                bool inBounds = nx >= 0 && nx < gridSize && nz >= 0 && nz < gridSize;
                result[i++] = inBounds ? tileTypes[nx, nz] : TileType.Terminal;
            }
        }
    }
}
