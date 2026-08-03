using UnityEngine;

public class GridManager : MonoBehaviour
{
    public int gridSize = 20;
    public float cellSize = 5f;
    public GameObject tilePrefab;

    [Header("Tile Materials")]
    public Material normalMaterial;
    public Material slipperyMaterial;
    public Material speedLimitedMaterial;
    public Material terminalMaterial;

    [Header("Tile Distribution (auto-normalized to sum to 1)")]
    [Range(0f, 1f)] public float probNormal       = 0.40f;
    [Range(0f, 1f)] public float probSpeedLimited  = 0.20f;
    [Range(0f, 1f)] public float probSlippery      = 0.20f;
    [Range(0f, 1f)] public float probTerminal      = 0.20f;

    [Header("Perlin Noise")]
    [Tooltip("Controls region size. Low (~0.1) = large blobs, high (~0.5) = fragmented patches. Lower this if you increase grid resolution.")]
    [Range(0.02f, 0.8f)] public float noiseScale = 0.15f;

    [Tooltip("When true (set by CarAgent during its bootstrap curriculum), every generated tile is " +
        "forced to TileType.Normal regardless of probXxx settings. GridManager has no ML-Agents " +
        "dependency and doesn't know why - it just does what it's told.")]
    public bool forceAllNormal = false;

    [HideInInspector] public TileType[,] tileTypes;
    private GridTile[,] tiles;
    private Vector3 gridOrigin;

    // Two independent noise offsets: one for passability, one for terrain type
    private float passOffsetX, passOffsetZ;
    private float terrainOffsetX, terrainOffsetZ;

    void Awake()
    {
        gridOrigin = new Vector3(-(gridSize * cellSize) / 2f, 0f, -(gridSize * cellSize) / 2f);
        tileTypes = new TileType[gridSize, gridSize];
        tiles = new GridTile[gridSize, gridSize];
        RandomiseNoiseOffset();
        GenerateGrid();
    }

    public void Regenerate()
    {
        gridOrigin = new Vector3(-(gridSize * cellSize) / 2f, 0f, -(gridSize * cellSize) / 2f);
        foreach (Transform child in transform)
            Destroy(child.gameObject);
        tileTypes = new TileType[gridSize, gridSize];
        tiles = new GridTile[gridSize, gridSize];
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
        float total = probNormal + probSpeedLimited + probSlippery + probTerminal;
        if (total <= 0f) total = 1f;

        // Noise 1 threshold: above this → Terminal (high noise = impassable blobs)
        float terminalThreshold = 1f - (probTerminal / total);

        // Noise 2 ordering: SpeedLimited → Normal → Slippery
        // Road sits in the middle so ice can border it directly on one side, grass on the other.
        float terrainTotal = probNormal + probSpeedLimited + probSlippery;
        if (terrainTotal <= 0f) terrainTotal = 1f;
        float tN  = probSpeedLimited / terrainTotal;          // below = SpeedLimited
        float tNS = tN + probNormal / terrainTotal;           // below = Normal, above = Slippery

        for (int x = 0; x < gridSize; x++)
        {
            for (int z = 0; z < gridSize; z++)
            {
                float passNoise = Mathf.PerlinNoise(
                    (x + passOffsetX) * noiseScale,
                    (z + passOffsetZ) * noiseScale
                );

                TileType type;
                if (forceAllNormal)
                {
                    type = TileType.Normal;
                }
                else if (passNoise >= terminalThreshold)
                {
                    type = TileType.Terminal;
                }
                else
                {
                    float terrainNoise = Mathf.PerlinNoise(
                        (x + terrainOffsetX) * noiseScale,
                        (z + terrainOffsetZ) * noiseScale
                    );

                    if      (terrainNoise < tN)  type = TileType.SpeedLimited;
                    else if (terrainNoise < tNS) type = TileType.Normal;
                    else                         type = TileType.Slippery;
                }

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
                if (mat != null)
                    go.GetComponent<Renderer>().material = mat;

                tileTypes[x, z] = type;
                tiles[x, z] = tile;
            }
        }
    }

    Material MaterialForType(TileType type)
    {
        switch (type)
        {
            case TileType.Normal:       return normalMaterial;
            case TileType.Slippery:     return slipperyMaterial;
            case TileType.SpeedLimited: return speedLimitedMaterial;
            case TileType.Terminal:     return terminalMaterial;
            default:                    return normalMaterial;
        }
    }

    // Unclamped: callers must bounds-check (see GetTileAt) so off-grid positions are distinguishable.
    public Vector2Int WorldToGrid(Vector3 worldPos)
    {
        int x = Mathf.FloorToInt((worldPos.x - gridOrigin.x) / cellSize);
        int z = Mathf.FloorToInt((worldPos.z - gridOrigin.z) / cellSize);
        return new Vector2Int(x, z);
    }

    public TileType GetTileAt(Vector3 worldPos)
    {
        var grid = tileTypes;
        if (grid == null) return TileType.Normal;
        var c = WorldToGrid(worldPos);
        if (c.x < 0 || c.x >= grid.GetLength(0) || c.y < 0 || c.y >= grid.GetLength(1))
            return TileType.Terminal; // off the map — treated as a hazard so the agent can see/avoid the edge
        return grid[c.x, c.y];
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
