"""Parameterised Voronoi maps: coloured regions separated by winding roads.

Pipeline
--------
1. Seeds are scattered, a few per colour, no two closer than `min_sep`.
2. Each seed grows a *ridge*: a random-walked spine thickened into a worm.
   Long thin ridges are what make the roads wind -- see `meander` below.
3. Roads are the true generalized Voronoi diagram of those ridges: every pixel
   goes to the nearest ridge, and the boundary of that assignment is drawn.
   Nothing about the roads is warped or smoothed afterwards.
4. Each region then expands into its own road-bounded cell, stopping `buffer`
   px short of the road. Roads are NOT recomputed after this -- fattening the
   regions first would make them convex and the roads would straighten out.

Main knobs
----------
meander   0 = straight roads, 1 = default, 2 = strongly winding
grow      how far a region spreads from its ridge when filling its cell
buffer    clear space kept between every region and the roads
counts    (reds, greens, blues); None picks 3-5 of each at random

Usage:
    python voronoi_highways.py
    python voronoi_highways.py --meander 1.8 --grow 50 --buffer 16
    python voronoi_highways.py --reds 6 --greens 6 --blues 6
"""
import argparse
import os
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
COLOR_NAMES = ("red", "green", "blue")
BG = (220, 220, 220)
ROAD = (0, 0, 0)
MARK_FILL, MARK_RING = (255, 255, 0), (40, 40, 40)


@dataclass
class Params:
    """Everything that controls a map. Defaults reproduce the tuned look."""
    size: int = 512

    # --- how many regions, and how far apart they start ------------------
    counts: Optional[Tuple[int, int, int]] = None   # None -> 3..5 of each
    min_sep: float = 70                             # min seed separation, px
    margin: int = 45                                # keep seeds off the border

    # --- how much the roads wind -----------------------------------------
    meander: float = 1.0        # 0 straight .. 2 strongly winding
    curl: Optional[float] = None    # advanced: override the three ridge
    spine: Optional[int] = None     # parameters that `meander` drives
    thick: Optional[float] = None
    reach: Optional[float] = None   # advanced: px a ridge may leave its cell
    step: float = 4                 # spine step length, px
    min_rad: float = 8              # spine stops when its cell gets tighter

    # --- how much of the map the regions cover ---------------------------
    grow: float = 34            # spread from the ridge; 0 leaves bare ridges
    buffer: float = 12          # clear space between a region and the roads

    # --- drawing ----------------------------------------------------------
    line: int = 10              # road width, px
    scaffold_gap: int = 10      # containment margin for the seed-Voronoi cells
    smooth: int = 3             # outline cleanup radius
    marks: int = 2              # yellow dots dropped on the roads
    mark_r: int = 11
    mark_sep: float = 120

    def ridge_shape(self):
        """Map the single `meander` scalar onto the three ridge parameters.

        At meander 0 the spine collapses to the seed itself and thickness stops
        varying, so every region is an identical disc centred on its seed. The
        bisector of two equal-radius discs is exactly the perpendicular bisector
        of their centres, so the roads come out as the seed-point Voronoi
        diagram, exactly. Raising meander lengthens and thins the ridge and lets
        its outline vary, which is what bends the boundaries.
        """
        m = max(0.0, self.meander)
        curl = self.curl if self.curl is not None else 0.5 * m
        spine = self.spine if self.spine is not None else int(round(48 * m))
        thick = self.thick if self.thick is not None else 17.0 / (1.0 + 0.35 * m)
        return curl, spine, thick

    def shape_noise(self):
        """How irregular a ridge outline gets. Zero at meander 0, rising to 2."""
        return 0.5 * min(1.0, max(0.0, self.meander) / 2.0)

    def spread(self):
        """How far a ridge may reach outside its own seed-Voronoi cell.

        Zero at meander 0, which keeps every region inside its cell and the
        roads exactly on the point-Voronoi. Above that, ridges interleave into
        neighbouring territory, which is what actually bends the boundaries --
        curl alone cannot, because a cell-confined ridge has nowhere to go.
        """
        if self.reach is not None:
            return self.reach
        return 45.0 * min(1.0, max(0.0, self.meander) / 2.0)

    def asymmetry(self):
        """How lopsided a ridge may be about its seed. Zero at meander 0.

        A ridge centred on its seed barely moves the boundary however long it
        is, which is why meander used to plateau. Letting one arm outrun the
        other offsets the region from its seed and shifts the bisector.
        """
        return 0.85 * min(1.0, max(0.0, self.meander) / 2.0)


@dataclass
class Result:
    image: Image.Image
    counts: Tuple[int, int, int]
    n_regions: int
    min_gap: float          # measured region-to-road clearance, px
    cover: float            # fraction of the map covered by regions
    wander: float           # mean px the roads run off the straight-line Voronoi
    marks: int


# ------------------------------------------------------------------ helpers
def disk(r):
    r = int(max(0, r))
    ky, kx = np.mgrid[-r:r + 1, -r:r + 1]
    return (ky ** 2 + kx ** 2) <= r ** 2


def thin_edges(label):
    e = np.zeros(label.shape, dtype=bool)
    e[:, :-1] |= label[:, :-1] != label[:, 1:]
    e[:, 1:] |= label[:, 1:] != label[:, :-1]
    e[:-1, :] |= label[:-1, :] != label[1:, :]
    e[1:, :] |= label[1:, :] != label[:-1, :]
    return e


def thick_edges(label, width):
    return ndimage.binary_dilation(thin_edges(label),
                                   structure=disk((width - 2) // 2))


def place_seeds(rng, p):
    """Poisson-ish seed scatter, relaxing the spacing if the map gets crowded."""
    counts = p.counts if p.counts is not None else tuple(
        int(rng.integers(3, 6)) for _ in COLORS)

    pts, cols, sep, tries = [], [], float(p.min_sep), 0
    for idx, n in enumerate(counts):
        placed = 0
        while placed < n:
            tries += 1
            if tries % 3000 == 0:
                sep *= 0.9          # crowded: accept tighter packing
            q = rng.integers(p.margin, p.size - p.margin, size=2)
            if all(np.hypot(*(q - r)) >= sep for r in pts):
                pts.append(q)
                cols.append(COLORS[idx])
                placed += 1
    return np.array(pts), cols, counts


def scaffold(rng, p, yy, xx):
    """Seed points and the straight-line Voronoi cells that contain the ridges.

    Scaffolding only: it guarantees regions cannot overlap, and is never drawn.
    """
    pts, cols, counts = place_seeds(rng, p)
    label = np.argmin(
        np.stack([(xx - px) ** 2 + (yy - py) ** 2 for px, py in pts]), axis=0)
    fillable = ndimage.distance_transform_edt(
        ~thick_edges(label, p.line)) > p.scaffold_gap
    return pts, cols, counts, label, fillable


def ridge(rng, sx, sy, cell, avail, rcap, p):
    """Grow one region: a random-walked spine thickened into a curving ridge.

    `rcap` is the largest radius that fits inside *every* cell on the map. It is
    what lets meander 0 produce identical discs: without a shared cap each
    region would be clipped to its own cell's clearance, the radii would differ,
    and the diagram would drift off the true point-Voronoi.
    """
    curl, spine_steps, thick = p.ridge_shape()
    thick = min(thick, rcap)
    size = cell.shape[0]

    spine = [(int(sy), int(sx))]
    heading = rng.uniform(0, 2 * np.pi)
    asym = p.asymmetry()
    arms = (spine_steps,
            int(spine_steps * rng.uniform(1.0 - asym, 1.0)))
    for back, steps in zip((0.0, np.pi), arms):
        x, y, th = float(sx), float(sy), heading + back
        for _ in range(steps):
            th += rng.normal(0, curl)
            nx, ny = x + p.step * np.cos(th), y + p.step * np.sin(th)
            iy, ix = int(round(ny)), int(round(nx))
            if not (0 <= iy < size and 0 <= ix < size):
                break
            if avail[iy, ix] < p.min_rad:
                break
            x, y = nx, ny
            spine.append((iy, ix))

    # thickness breathes along the ridge so it is not a uniform sausage;
    # the amplitude is zero at meander 0, leaving a perfectly regular disc
    amp = p.shape_noise()
    # high harmonics fade in with meander, so the outline gains fine detail
    # rather than merely swinging harder at one wavelength
    hi = min(1.0, max(0.0, p.meander - 0.5) / 1.5)
    freqs = np.array([1.0, 2.3, 3.7, 6.1, 9.3])
    wts = np.array([1.0, 1.0, 1.0, hi, hi])
    t = np.linspace(0, 1, len(spine))
    ph = rng.uniform(0, 2 * np.pi, size=freqs.size)
    var = sum(w * np.sin(2 * np.pi * f * t + q)
              for f, w, q in zip(freqs, wts, ph))
    var = (1.0 - amp) + amp * (var / wts.sum())

    m = np.zeros_like(cell)
    for k, (iy, ix) in enumerate(spine):
        r = int(max(4, min(avail[iy, ix] * 0.85, thick) * var[k]))
        y0, y1 = max(0, iy - r), min(size, iy + r + 1)
        x0, x1 = max(0, ix - r), min(size, ix + r + 1)
        d = disk(r)
        m[y0:y1, x0:x1] |= d[y0 - (iy - r):y1 - (iy - r),
                             x0 - (ix - r):x1 - (ix - r)]

    st = disk(p.smooth)
    m = ndimage.binary_closing(m, structure=st)
    m = ndimage.binary_opening(m, structure=st)
    return m & cell


def voronoi_roads(region_id, line):
    """True generalized Voronoi of the regions; also returns the cell map."""
    _, (iy, ix) = ndimage.distance_transform_edt(region_id == 0,
                                                 return_indices=True)
    cells = region_id[iy, ix]
    return thick_edges(cells, line), cells


def expand(region_id, cells, roads, n, p):
    """Fatten each ridge into its own cell, stopping `buffer` px off the roads."""
    if p.grow <= 0:
        return region_id
    room = ndimage.distance_transform_edt(~roads) > p.buffer
    st = disk(p.smooth)

    grown = np.zeros_like(region_id)
    for i in range(1, n + 1):
        reach = ndimage.distance_transform_edt(region_id != i)
        m = (reach <= p.grow) & (cells == i) & room
        m = ndimage.binary_closing(m, structure=st)
        m = ndimage.binary_opening(m, structure=st)
        # closing can bleed past the caps, so re-apply them after smoothing
        grown[m & (cells == i) & room] = i
    return grown


# ------------------------------------------------------------------ generate
def generate(seed, p: Params) -> Result:
    size = p.size
    yy, xx = np.mgrid[0:size, 0:size]
    rng = np.random.default_rng(seed)

    pts, cols, counts, cell_label, fillable = scaffold(rng, p, yy, xx)

    # radius that fits inside every cell, so meander 0 yields identical discs
    masks = [(cell_label == i) & fillable for i in range(len(pts))]
    rcap = min(ndimage.distance_transform_edt(m)[sy, sx]
               for m, (sx, sy) in zip(masks, pts)) * 0.85

    # Ridges are placed one at a time. Each may stray `spread` px outside its
    # own cell but must stay `keep_out` clear of everything already placed, so
    # they interleave without ever colliding -- and a road always fits between.
    spread = p.spread()
    keep_out = p.line + 2 * p.buffer
    order = rng.permutation(len(pts))
    placed = np.zeros((size, size), dtype=bool)
    region_id = np.zeros((size, size), dtype=np.int32)

    for i in order:
        sx, sy = pts[i]
        allowed = masks[i]
        if spread > 0:
            allowed = ndimage.distance_transform_edt(~allowed) <= spread
        if placed.any():
            allowed = allowed & (
                ndimage.distance_transform_edt(~placed) > keep_out)

        avail = ndimage.distance_transform_edt(allowed)
        if avail[sy, sx] < p.min_rad:
            # a neighbour's ridge crowded this seed; restart from the roomiest
            # spot still inside the seed's own cell rather than drop the region
            room = np.where(masks[i], avail, 0)
            if room.max() < 4:
                continue
            sy, sx = np.unravel_index(room.argmax(), room.shape)

        m = ridge(rng, sx, sy, allowed, avail, rcap, p)
        region_id[m] = i + 1
        placed |= m

    roads, cells = voronoi_roads(region_id, p.line)
    region_id = expand(region_id, cells, roads, len(pts), p)

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

    # How far the roads run from the straight-line Voronoi of the same seeds.
    # Road *length* is a poor meander measure -- a gentle curve is barely longer
    # than a straight line, and topology changes swamp the difference. Distance
    # off the straight baseline separates the settings cleanly.
    baseline = thin_edges(cell_label)
    wander = ndimage.distance_transform_edt(~baseline)[thin_edges(cells)].mean()

    return Result(img, counts, len(pts), float(gap),
                  float(occupied.mean()), float(wander), len(chosen))


# ---------------------------------------------------------------------- CLI
def params_from_args(a) -> Params:
    counts = None
    if a.reds is not None or a.greens is not None or a.blues is not None:
        counts = (a.reds if a.reds is not None else 4,
                  a.greens if a.greens is not None else 4,
                  a.blues if a.blues is not None else 4)
    return Params(
        size=a.size, counts=counts, min_sep=a.min_sep,
        meander=a.meander, curl=a.curl, spine=a.spine, thick=a.thick,
        grow=a.grow, buffer=a.buffer, line=a.line, smooth=a.smooth,
        marks=a.marks)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=".")
    p.add_argument("--rows", type=int, default=2)
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--size", type=int, default=512)

    g = p.add_argument_group("regions")
    g.add_argument("--reds", type=int, help="number of red regions")
    g.add_argument("--greens", type=int, help="number of green regions")
    g.add_argument("--blues", type=int, help="number of blue regions")
    g.add_argument("--min-sep", type=float, default=70,
                   help="minimum seed separation in px")

    g = p.add_argument_group("road meander")
    g.add_argument("--meander", type=float, default=1.0,
                   help="0 straight, 1 default, 2 strongly winding")
    g.add_argument("--curl", type=float, help="advanced: spine turn per step")
    g.add_argument("--spine", type=int, help="advanced: spine steps per side")
    g.add_argument("--thick", type=float, help="advanced: ridge half-width")

    g = p.add_argument_group("region fill")
    g.add_argument("--grow", type=float, default=34,
                   help="spread from the ridge; 0 leaves bare ridges, "
                        "60+ fills each cell completely")
    g.add_argument("--buffer", type=float, default=12,
                   help="clear space between a region and the roads")

    g = p.add_argument_group("drawing")
    g.add_argument("--line", type=int, default=10, help="road width in px")
    g.add_argument("--smooth", type=int, default=3)
    g.add_argument("--marks", type=int, default=2, help="yellow dots per map")
    g.add_argument("--gutter", type=int, default=12)
    a = p.parse_args()

    par = params_from_args(a)
    tiles = os.path.join(a.out, "maps_highways")
    os.makedirs(tiles, exist_ok=True)

    imgs = []
    for i in range(a.rows * a.cols):
        r = generate(a.seed + i, par)
        r.image.save(os.path.join(tiles, f"highway_{i + 1}.png"))
        imgs.append(r.image)
        print(f"map {i + 1} (seed {a.seed + i}): "
              f"R={r.counts[0]} G={r.counts[1]} B={r.counts[2]} "
              f"({r.n_regions} regions) | gap {r.min_gap:.1f}px | "
              f"cover {r.cover:.0%} | tortuosity {r.tortuosity:.2f}")

    s, gt = a.size, a.gutter
    W = a.cols * s + (a.cols + 1) * gt
    H = a.rows * s + (a.rows + 1) * gt
    grid = Image.new("RGB", (W, H), (255, 255, 255))
    for i, img in enumerate(imgs):
        r, c = divmod(i, a.cols)
        grid.paste(img, (gt + c * (s + gt), gt + r * (s + gt)))
    out = os.path.join(a.out, f"grid_{a.rows}x{a.cols}_highways.png")
    grid.save(out)
    print(f"\ngrid: {W}x{H} -> {out}  (tiles in {tiles})")


if __name__ == "__main__":
    main()
