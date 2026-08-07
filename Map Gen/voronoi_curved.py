"""Curved-road Voronoi maps: perturb the straight diagram, then fill the cells.

Pipeline (reverse of the earlier ridge approach)
------------------------------------------------
1. Scatter seeds, a few per colour, and take their ordinary point-Voronoi
   diagram. Seeds are mirrored across the four borders so every interior edge
   is finite and terminates on the image boundary.
2. Turn each straight Voronoi edge into a gentle curve: drop a few anchors along
   it, offset each one perpendicular to the edge by a random amount, and run a
   cubic spline through them. Edge *endpoints* are left untouched, so junctions
   stay put and the road network remains connected and closed.
3. Rasterise the splines as roads. Their complement is the set of cells.
4. Grow an organic blob inside each cell, holding `buffer` px off the roads.

Because the offset is applied to the straight diagram, `meander = 0` reproduces
the exact point-Voronoi, and curvature rises monotonically from there -- the
amplitude is the parameter, not an emergent property of some blob shape.

Usage:
    python voronoi_curved.py
    python voronoi_curved.py --meander 0.9 --anchors 3 --fill 0.8
    python voronoi_curved.py --reds 6 --greens 3 --blues 3 --buffer 16
"""
import argparse
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.interpolate import splev, splprep
from scipy.spatial import Voronoi

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
BG = (220, 220, 220)
ROAD = (0, 0, 0)
MARK_FILL, MARK_RING = (255, 255, 0), (40, 40, 40)
NBINS = 180


@dataclass
class Params:
    size: int = 512

    # --- how many regions ------------------------------------------------
    counts: Optional[Tuple[int, int, int]] = None   # None -> 3..5 of each
    min_sep: float = 70
    margin: int = 45

    # --- how much the roads curve ----------------------------------------
    meander: float = 0.6        # 0 = exact straight Voronoi, 1 = strongly bent
    anchors: int = 2            # spline anchors per edge; more = tighter wiggle
    max_off: float = 0.45       # offset cap, as a fraction of seed clearance

    # --- how much of each cell the region fills --------------------------
    fill: float = 0.92          # 1.0 fills the cell right up to the buffer
    wobble: float = 0.18        # irregularity of the blob outline
    buffer: float = 12          # clear space between a region and the roads

    # --- drawing ----------------------------------------------------------
    line: int = 10
    smooth: int = 3
    marks: int = 2
    mark_r: int = 11
    mark_sep: float = 120


@dataclass
class Result:
    image: Image.Image
    counts: Tuple[int, int, int]
    n_regions: int
    min_gap: float
    cover: float
    wander: float        # mean px the curved roads sit off the straight ones
    marks: int


def disk(r):
    r = int(max(0, r))
    ky, kx = np.mgrid[-r:r + 1, -r:r + 1]
    return (ky ** 2 + kx ** 2) <= r ** 2


# ------------------------------------------------------------------- seeds
def place_seeds(rng, p):
    counts = p.counts if p.counts is not None else tuple(
        int(rng.integers(3, 6)) for _ in COLORS)
    pts, cols, sep, tries = [], [], float(p.min_sep), 0
    for idx, n in enumerate(counts):
        placed = 0
        while placed < n:
            tries += 1
            if tries % 3000 == 0:
                sep *= 0.9
            q = rng.integers(p.margin, p.size - p.margin, size=2)
            if all(np.hypot(*(q - r)) >= sep for r in pts):
                pts.append(q)
                cols.append(COLORS[idx])
                placed += 1
    return np.array(pts, dtype=float), cols, counts


# ------------------------------------------------------------------- edges
def voronoi_edges(pts, size):
    """Interior Voronoi edges as (A, B) endpoint pairs.

    The seeds are mirrored across all four borders. That makes every edge
    between two real seeds finite and ends it exactly on the image boundary,
    which closes the border cells -- without it the outer cells all leak into
    one another around the edge of the frame.
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
        if a < n and b < n and v1 >= 0 and v2 >= 0:
            edges.append((vor.vertices[v1].copy(), vor.vertices[v2].copy()))
    return edges


def curve_edge(A, B, pts, p, rng):
    """Spline a straight edge into a gentle curve, endpoints pinned.

    Each anchor's offset is capped at a fraction of the distance from that
    point to the nearest seed, so a curve can never bend far enough to swallow
    a seed or cross into the wrong cell.
    """
    span = B - A
    L = float(np.hypot(*span))
    if L < 2:
        return np.array([A, B])

    d = span / L
    nrm = np.array([-d[1], d[0]])

    ctrl = [A]
    for k in range(1, p.anchors + 1):
        t = k / (p.anchors + 1)
        base = A + t * span
        near = float(np.min(np.hypot(*(pts - base).T)))
        cap = min(p.max_off * near, 0.40 * L)
        ctrl.append(base + nrm * (p.meander * cap * rng.uniform(-1, 1)))
    ctrl.append(B)

    ctrl = np.array(ctrl)
    deg = min(3, len(ctrl) - 1)
    try:
        tck, _ = splprep([ctrl[:, 0], ctrl[:, 1]], s=0, k=deg)
    except (TypeError, ValueError):
        return ctrl
    x, y = splev(np.linspace(0, 1, max(int(L), 12)), tck)
    return np.column_stack([x, y])


def draw_roads(polys, size, line):
    img = Image.new("1", (size, size), 0)
    d = ImageDraw.Draw(img)
    r = line // 2
    for poly in polys:
        pts = [tuple(map(float, q)) for q in poly]
        if len(pts) > 1:
            d.line(pts, fill=1, width=line, joint="curve")
        for q in (pts[0], pts[-1]):            # round off the junctions
            d.ellipse([q[0] - r, q[1] - r, q[0] + r, q[1] + r], fill=1)
    return np.array(img, dtype=bool)


# ------------------------------------------------------------------- blobs
def blob(rng, sx, sy, cell, yy, xx, p):
    """Organic blob filling a share of its cell, measured per direction."""
    rho = np.hypot(xx - sx, yy - sy)
    th = np.arctan2(yy - sy, xx - sx)
    bins = (((th + np.pi) / (2 * np.pi)) * NBINS).astype(int) % NBINS

    big = float(rho.max())
    reach = np.full(NBINS, big, dtype=float)
    out = ~cell
    np.minimum.at(reach, bins[out], rho[out])
    reach = ndimage.minimum_filter1d(reach, size=7, mode="wrap")
    reach = ndimage.uniform_filter1d(reach, size=5, mode="wrap")

    ks = np.array([2, 3, 4, 5])
    amps = rng.uniform(0.35, 1.0, size=ks.size)
    ph = rng.uniform(0, 2 * np.pi, size=ks.size)
    wob = sum(a * np.sin(k * th + q) for k, a, q in zip(ks, amps, ph))
    wob /= amps.sum()

    prof = reach[bins] * p.fill * (1.0 - p.wobble + p.wobble * wob)
    m = (rho <= prof) & cell
    st = disk(p.smooth)
    m = ndimage.binary_closing(m, structure=st)
    m = ndimage.binary_opening(m, structure=st)
    return m & cell


# ---------------------------------------------------------------- generate
def generate(seed, p: Params) -> Result:
    size = p.size
    yy, xx = np.mgrid[0:size, 0:size]
    rng = np.random.default_rng(seed)

    pts, cols, counts = place_seeds(rng, p)
    edges = voronoi_edges(pts, size)

    straight = draw_roads([np.array([A, B]) for A, B in edges], size, p.line)
    roads = draw_roads([curve_edge(A, B, pts, p, rng) for A, B in edges],
                       size, p.line)

    # cells are simply the connected components left over by the road network
    lab, _ = ndimage.label(~roads)
    free = ndimage.distance_transform_edt(~roads) > p.buffer

    region_id = np.zeros((size, size), dtype=np.int32)
    for i, (sx, sy) in enumerate(pts):
        ix, iy = int(round(sx)), int(round(sy))
        comp = lab[iy, ix]
        if comp == 0:
            continue
        cell = (lab == comp) & free
        if not cell[iy, ix]:
            continue
        region_id[blob(rng, ix, iy, cell, yy, xx, p)] = i + 1

    arr = np.full((size, size, 3), BG, dtype=np.uint8)
    for i, c in enumerate(cols):
        arr[region_id == i + 1] = c
    arr[roads] = ROAD

    img = Image.fromarray(arr)
    d = ImageDraw.Draw(img)

    pad = p.mark_r + p.line
    cand = np.argwhere(roads)
    cand = cand[(cand[:, 0] > pad) & (cand[:, 0] < size - pad) &
                (cand[:, 1] > pad) & (cand[:, 1] < size - pad)]
    chosen, guard = [], 0
    while len(chosen) < p.marks and len(cand) and guard < 20000:
        guard += 1
        ry, rx = cand[rng.integers(len(cand))]
        if all(np.hypot(ry - cy, rx - cx) > p.mark_sep for cy, cx in chosen):
            chosen.append((ry, rx))
    for cy, cx in chosen:
        d.ellipse([cx - p.mark_r, cy - p.mark_r, cx + p.mark_r, cy + p.mark_r],
                  fill=MARK_FILL, outline=MARK_RING, width=2)

    occupied = region_id > 0
    gap = (ndimage.distance_transform_edt(~roads)[occupied].min()
           if occupied.any() else float("nan"))
    wander = ndimage.distance_transform_edt(~straight)[roads].mean()
    n = len(np.unique(region_id)) - 1

    return Result(img, counts, n, float(gap), float(occupied.mean()),
                  float(wander), len(chosen))


# --------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=".")
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--size", type=int, default=512)

    g = ap.add_argument_group("regions")
    g.add_argument("--reds", type=int)
    g.add_argument("--greens", type=int)
    g.add_argument("--blues", type=int)
    g.add_argument("--min-sep", type=float, default=70)

    g = ap.add_argument_group("road curvature")
    g.add_argument("--meander", type=float, default=0.6,
                   help="0 = exact straight Voronoi, 1 = strongly bent")
    g.add_argument("--anchors", type=int, default=2,
                   help="spline anchors per edge; more = tighter wiggle")
    g.add_argument("--max-off", type=float, default=0.45,
                   help="offset cap as a fraction of seed clearance")

    g = ap.add_argument_group("region fill")
    g.add_argument("--fill", type=float, default=0.92,
                   help="share of the cell a region occupies")
    g.add_argument("--wobble", type=float, default=0.18,
                   help="irregularity of the blob outline")
    g.add_argument("--buffer", type=float, default=12,
                   help="clear space between a region and the roads")

    g = ap.add_argument_group("drawing")
    g.add_argument("--line", type=int, default=10)
    g.add_argument("--marks", type=int, default=2)
    g.add_argument("--gutter", type=int, default=12)
    a = ap.parse_args()

    counts = None
    if a.reds is not None or a.greens is not None or a.blues is not None:
        counts = (a.reds or 4, a.greens or 4, a.blues or 4)
    p = Params(size=a.size, counts=counts, min_sep=a.min_sep,
               meander=a.meander, anchors=a.anchors, max_off=a.max_off,
               fill=a.fill, wobble=a.wobble, buffer=a.buffer,
               line=a.line, marks=a.marks)

    tiles = os.path.join(a.out, "maps_curved")
    os.makedirs(tiles, exist_ok=True)

    imgs = []
    for i in range(a.rows * a.cols):
        r = generate(a.seed + i, p)
        r.image.save(os.path.join(tiles, f"curved_{i + 1}.png"))
        imgs.append(r.image)
        print(f"map {i + 1} (seed {a.seed + i}): R={r.counts[0]} G={r.counts[1]} "
              f"B={r.counts[2]} ({r.n_regions} regions) | gap {r.min_gap:.1f}px | "
              f"cover {r.cover:.0%} | wander {r.wander:.1f}px")

    s, gt = a.size, a.gutter
    W, H = a.cols * s + (a.cols + 1) * gt, a.rows * s + (a.rows + 1) * gt
    grid = Image.new("RGB", (W, H), (255, 255, 255))
    for i, img in enumerate(imgs):
        r, c = divmod(i, a.cols)
        grid.paste(img, (gt + c * (s + gt), gt + r * (s + gt)))
    out = os.path.join(a.out, f"grid_{a.rows}x{a.cols}_curved.png")
    grid.save(out)
    print(f"\ngrid: {W}x{H} -> {out}  (tiles in {tiles})")


if __name__ == "__main__":
    main()
