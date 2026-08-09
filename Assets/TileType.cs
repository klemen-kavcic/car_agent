// Normal was merged into Asphalt - both were physics-identical (no special case in
// car_component.cs's ApplyTileEffects, i.e. full speed/traction) and played the exact same
// "default safe ground" role, just under different mapSources (Perlin vs Voronoi).
public enum TileType { Slippery = 0, SpeedLimited = 1, Terminal = 2, Asphalt = 3, Gravel = 4 }
