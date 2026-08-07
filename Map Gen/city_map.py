"""City-block maps: rotated grid districts, ring roads, and winding highways.

Layout
------
* Several rectangular **districts**, each a rotated R x C lattice of city blocks
  at its own random angle.
* Green and blue seeds scattered in the **gaps** between districts.

Roads
-----
* **Local streets** are drawn explicitly from the district lattice rather than
  derived from the Voronoi diagram. That is what makes the *peripheral ring*
  possible: the outermost grid lines close the district, which no interior
  Voronoi edge ever does. Street ends also **reach out to a nearby highway**
  when one lies within `extend` px along their own direction, so districts
  connect to the network instead of stopping dead.
* **Highways** are the Voronoi edges that are not internal to a district --
  district-to-district, or anything touching a green/blue seed -- splined into
  curves with their endpoints pinned, and drawn thick.

Regions
-------
Red blocks are drawn as inset rectangles, so they stay crisp and rectangular.
Only green and blue cells get organic blobs.

Usage:
    python city_map.py
    python city_map.py --districts 4 --rotate 40 --extend 120
    python city_map.py --spacing 64 --meander 0.9 --greens 7 --blues 5
"""
import argparse
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import Voronoi

from voronoi_curved import blob, curve_edge, draw_roads

RED, GREEN, BLUE = (255, 0, 0), (0, 255, 0), (0, 0, 255)
BG = (220, 220, 220)
ROAD = (0, 0, 0)


@dataclass
class Params:
    size: int = 640

    # --- districts of red blocks ------------------------------------------
    districts: int = 3
    grid_min: int = 2               # lattice rows/cols drawn from
    grid_max: int = 4               # [grid_min, grid_max]
    spacing: float = 58             # lattice pitch -> city block size
    rotate: float = 40.0            # max district rotation, degrees
    jitter: float = 0.75            # see note in build_districts
    district_gap: float = 60        # keep district footprints this far apart
    extend: float = 110             # how far a street will reach for a highway
    margin: int = 30

    # --- outskirt seeds ----------------------------------------------------
    greens: int = 6
    blues: int = 6
    gap_margin: float = 30          # keep them clear of the district footprints
    min_sep: float = 62             # ... and of each other

    # --- highway curvature (local streets are always straight) -------------
    meander: float = 0.7
    anchors: int = 2
    max_off: float = 0.42

    # --- fills -------------------------------------------------------------
    fill: float = 0.92              # green/blue blob share of its cell
    wobble: float = 0.18            # green/blue outline irregularity
    buffer: float = 9               # clear space between any region and a road

    # --- drawing -----------------------------------------------------------
    local_line: int = 6
    highway_line: int = 13
    line: int = 13                  # used by the imported spline rasteriser
    smooth: int = 3


@dataclass
class District:
    center: np.ndarray
    angle: float
    rows: int
    cols: int
    spacing: float

    def _to_world(self, local):
        c, s = np.cos(self.angle), np.sin(self.angle)
        R = np.array([[c, -s], [s, c]])
        return local @ R.T + self.center

    def block_centers(self):
        gy, gx = np.mgrid[0:self.rows, 0:self.cols]
        local = np.column_stack([
            (gx.ravel() - (self.cols - 1) / 2) * self.spacing,
            (gy.ravel() - (self.rows - 1) / 2) * self.spacing]).astype(float)
        return self._to_world(local)

    def street_lines(self):
        """Grid lines including the outermost ones, which form the ring road."""
        X, Y = self.cols * self.spacing / 2, self.rows * self.spacing / 2
        segs = []
        for k in range(self.cols + 1):
            x = -X + k * self.spacing
            segs.append(self._to_world(np.array([[x, -Y], [x, Y]])))
        for m in range(self.rows + 1):
            y = -Y + m * self.spacing
            segs.append(self._to_world(np.array([[-X, y], [X, y]])))
        return segs

    def block_quads(self, inset):
        """Each block as an inset quad, so it stays a crisp rectangle."""
        X, Y = self.cols * self.spacing / 2, self.rows * self.spacing / 2
        h = self.spacing / 2 - inset
        quads = []
        for i in range(self.rows):
            for j in range(self.cols):
                cx = -X + (j + 0.5) * self.spacing
                cy = -Y + (i + 0.5) * self.spacing
                local = np.array([[cx - h, cy - h], [cx + h, cy - h],
                                  [cx + h, cy + h], [cx - h, cy + h]])
                quads.append(self._to_world(local))
        return quads

    def corners(self):
        X, Y = self.cols * self.spacing / 2, self.rows * self.spacing / 2
        return self._to_world(np.array([[-X, -Y], [X, -Y], [X, Y], [-X, Y]]))

    def bbox(self):
        c = self.corners()
        return (c[:, 0].min(), c[:, 1].min(), c[:, 0].max(), c[:, 1].max())


# --------------------------------------------------------------- placement
def _clash(a, b, gap):
    return not (a[2] + gap < b[0] or b[2] + gap < a[0] or
                a[3] + gap < b[1] or b[3] + gap < a[1])


def build_districts(rng, p):
    out: List[District] = []
    for _ in range(p.districts):
        for _ in range(500):
            d = District(np.zeros(2),
                         np.radians(rng.uniform(-p.rotate, p.rotate)),
                         int(rng.integers(p.grid_min, p.grid_max + 1)),
                         int(rng.integers(p.grid_min, p.grid_max + 1)),
                         p.spacing)
            probe = d.bbox()
            w, h = probe[2] - probe[0], probe[3] - probe[1]
            if w > p.size - 2 * p.margin or h > p.size - 2 * p.margin:
                continue
            d.center = np.array([
                rng.uniform(p.margin + w / 2, p.size - p.margin - w / 2),
                rng.uniform(p.margin + h / 2, p.size - p.margin - h / 2)])
            if any(_clash(d.bbox(), q.bbox(), p.district_gap) for q in out):
                continue
            out.append(d)
            break
    return out


def build_seeds(rng, p, districts):
    """Voronoi seeds: jittered block centres, plus green/blue in the gaps.

    The jitter only touches the seeds handed to Qhull -- a perfect lattice puts
    four seeds on a common circle, which is degenerate and yields zero-length
    edges. The streets are drawn from the ideal lattice, so they stay exact.
    """
    pts, cols, owner = [], [], []
    for di, d in enumerate(districts):
        for q in d.block_centers():
            pts.append(q + rng.normal(0, p.jitter, size=2))
            cols.append(RED)
            owner.append(di)

    boxes = [d.bbox() for d in districts]
    for colour, n in ((GREEN, p.greens), (BLUE, p.blues)):
        placed, tries = 0, 0
        while placed < n and tries < 20000:
            tries += 1
            q = rng.uniform(p.margin, p.size - p.margin, size=2)
            if any(b[0] - p.gap_margin < q[0] < b[2] + p.gap_margin and
                   b[1] - p.gap_margin < q[1] < b[3] + p.gap_margin
                   for b in boxes):
                continue
            if any(np.hypot(*(q - r)) < p.min_sep for r in pts):
                continue
            pts.append(q)
            cols.append(colour)
            owner.append(-1)
            placed += 1
    return np.array(pts, float), cols, np.array(owner)


# ------------------------------------------------------------------- edges
def highway_edges(pts, owner, size):
    """Voronoi edges that are not internal to a single district.

    Seeds are mirrored across all four borders so every interior edge is finite
    and terminates on the image boundary, closing the outer cells.
    """
    n = len(pts)
    mir = []
    for axis, val in ((0, 0.0), (0, 2.0 * size), (1, 0.0), (1, 2.0 * size)):
        m = pts.copy()
        m[:, axis] = val - m[:, axis]
        mir.append(m)
    vor = Voronoi(np.vstack([pts] + mir))

    edges = []
    for (a, b), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        if not (a < n and b < n and v1 >= 0 and v2 >= 0):
            continue
        if owner[a] >= 0 and owner[a] == owner[b]:
            continue                      # internal street, drawn from lattice
        edges.append((vor.vertices[v1].copy(), vor.vertices[v2].copy()))
    return edges


def reach_for_highway(pt, direction, highways, limit, size):
    """March outward from a street end; return the highway hit, or None."""
    for t in range(4, int(limit) + 1, 2):
        q = pt + direction * t
        ix, iy = int(round(q[0])), int(round(q[1]))
        if not (0 <= ix < size and 0 <= iy < size):
            return None
        if highways[iy, ix]:
            return q
    return None


def street_segments(districts, highways, p, size):
    """District grid lines, each end extended to a highway when one is close."""
    segs, hits = [], 0
    for d in districts:
        for seg in d.street_lines():
            A, B = seg[0].copy(), seg[1].copy()
            u = B - A
            u = u / max(np.hypot(*u), 1e-9)
            far = reach_for_highway(B, u, highways, p.extend, size)
            if far is not None:
                B, hits = far, hits + 1
            near = reach_for_highway(A, -u, highways, p.extend, size)
            if near is not None:
                A, hits = near, hits + 1
            segs.append(np.array([A, B]))
    return segs, hits


# ---------------------------------------------------------------- generate
@dataclass
class Result:
    image: Image.Image
    n_districts: int
    n_blocks: int
    n_green: int
    n_blue: int
    n_streets: int
    n_highways: int
    extensions: int
    min_gap: float
    cover: float


def generate(seed, p: Params) -> Result:
    size = p.size
    yy, xx = np.mgrid[0:size, 0:size]
    rng = np.random.default_rng(seed)

    districts = build_districts(rng, p)
    pts, cols, owner = build_seeds(rng, p, districts)

    hw = highway_edges(pts, owner, size)
    curved = [curve_edge(A, B, pts, p, rng) for A, B in hw]
    highway_mask = draw_roads(curved, size, p.highway_line)

    # streets are laid after the highways exist, so they can find them
    segs, extensions = street_segments(districts, highway_mask, p, size)
    local_mask = draw_roads(segs, size, p.local_line)
    roads = local_mask | highway_mask

    free = ndimage.distance_transform_edt(~roads) > p.buffer

    # red blocks: inset quads, then clipped by anything a road actually covers
    blocks = Image.new("1", (size, size), 0)
    bd = ImageDraw.Draw(blocks)
    n_blocks = 0
    for d in districts:
        for quad in d.block_quads(p.local_line / 2 + p.buffer):
            bd.polygon([tuple(map(float, q)) for q in quad], fill=1)
            n_blocks += 1
    red_mask = np.array(blocks, dtype=bool) & free

    arr = np.full((size, size, 3), BG, dtype=np.uint8)
    arr[red_mask] = RED

    # green/blue: organic blobs filling their road-bounded cell
    lab, _ = ndimage.label(~roads)
    occupied = red_mask.copy()
    for i, q in enumerate(pts):
        if owner[i] >= 0:
            continue
        ix, iy = int(round(q[0])), int(round(q[1]))
        if not (0 <= ix < size and 0 <= iy < size):
            continue
        comp = lab[iy, ix]
        if comp == 0:
            continue
        cell = (lab == comp) & free & ~red_mask
        if not cell[iy, ix]:
            continue
        m = blob(rng, ix, iy, cell, yy, xx, p)
        arr[m] = cols[i]
        occupied |= m

    arr[roads] = ROAD
    gap = (ndimage.distance_transform_edt(~roads)[occupied].min()
           if occupied.any() else float("nan"))

    return Result(Image.fromarray(arr), len(districts), n_blocks,
                  sum(1 for c in cols if c == GREEN),
                  sum(1 for c in cols if c == BLUE),
                  len(segs), len(hw), extensions,
                  float(gap), float(occupied.mean()))


# --------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=".")
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--cols", type=int, default=2)
    ap.add_argument("--seed", type=int, default=300)
    ap.add_argument("--size", type=int, default=640)

    g = ap.add_argument_group("districts (red)")
    g.add_argument("--districts", type=int, default=3)
    g.add_argument("--grid-min", type=int, default=2)
    g.add_argument("--grid-max", type=int, default=4)
    g.add_argument("--spacing", type=float, default=58, help="city block pitch")
    g.add_argument("--rotate", type=float, default=40.0,
                   help="max district rotation in degrees")
    g.add_argument("--district-gap", type=float, default=60)
    g.add_argument("--extend", type=float, default=110,
                   help="how far a street reaches for a nearby highway")

    g = ap.add_argument_group("outskirts (green/blue)")
    g.add_argument("--greens", type=int, default=6)
    g.add_argument("--blues", type=int, default=6)
    g.add_argument("--gap-margin", type=float, default=30)

    g = ap.add_argument_group("highways")
    g.add_argument("--meander", type=float, default=0.7)
    g.add_argument("--anchors", type=int, default=2)

    g = ap.add_argument_group("fills and drawing")
    g.add_argument("--fill", type=float, default=0.92)
    g.add_argument("--wobble", type=float, default=0.18)
    g.add_argument("--buffer", type=float, default=9)
    g.add_argument("--local-line", type=int, default=6)
    g.add_argument("--highway-line", type=int, default=13)
    g.add_argument("--gutter", type=int, default=12)
    a = ap.parse_args()

    p = Params(size=a.size, districts=a.districts, grid_min=a.grid_min,
               grid_max=a.grid_max, spacing=a.spacing, rotate=a.rotate,
               district_gap=a.district_gap, extend=a.extend,
               greens=a.greens, blues=a.blues, gap_margin=a.gap_margin,
               meander=a.meander, anchors=a.anchors, fill=a.fill,
               wobble=a.wobble, buffer=a.buffer, local_line=a.local_line,
               highway_line=a.highway_line, line=a.highway_line)

    tiles = os.path.join(a.out, "maps_city")
    os.makedirs(tiles, exist_ok=True)

    imgs = []
    for i in range(a.rows * a.cols):
        r = generate(a.seed + i, p)
        r.image.save(os.path.join(tiles, f"city_{i + 1}.png"))
        imgs.append(r.image)
        print(f"map {i + 1} (seed {a.seed + i}): {r.n_districts} districts / "
              f"{r.n_blocks} blocks | {r.n_green}G {r.n_blue}B | "
              f"{r.n_streets} streets ({r.extensions} extended) + "
              f"{r.n_highways} highways | gap {r.min_gap:.1f}px | "
              f"cover {r.cover:.0%}")

    s, gt = a.size, a.gutter
    W, H = a.cols * s + (a.cols + 1) * gt, a.rows * s + (a.rows + 1) * gt
    grid = Image.new("RGB", (W, H), (255, 255, 255))
    for i, img in enumerate(imgs):
        r, c = divmod(i, a.cols)
        grid.paste(img, (gt + c * (s + gt), gt + r * (s + gt)))
    out = os.path.join(a.out, f"grid_{a.rows}x{a.cols}_city.png")
    grid.save(out)
    print(f"\ngrid: {W}x{H} -> {out}  (tiles in {tiles})")


if __name__ == "__main__":
    main()
