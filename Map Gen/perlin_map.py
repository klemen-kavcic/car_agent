"""Perlin-region maps with organic Voronoi roads.

Regions come from thresholding a fractal Perlin noise field, so their shapes are
noise-derived rather than grown from seed points. The roads reuse the generalized
(region) Voronoi logic: every pixel is assigned to the nearest region, and the
boundaries of that assignment become the black lines, so roads wind between the
regions instead of running straight.

Usage:
    python perlin_map.py                      # 2x4 grid of 8 maps
    python perlin_map.py --rows 3 --cols 3 --seed 42
    python perlin_map.py --size 768 --octaves 5 --land 0.42
"""
import argparse
import os

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
BG = (220, 220, 220)
ROAD = (0, 0, 0)
MARK_FILL, MARK_RING = (255, 255, 0), (40, 40, 40)


# --------------------------------------------------------------- perlin noise
def perlin(shape, res, rng):
    """Classic gradient noise on a res x res lattice, bilinearly faded."""
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:1 / d[0], 0:res[1]:1 / d[1]].transpose(1, 2, 0) % 1

    angles = 2 * np.pi * rng.random((res[0] + 1, res[1] + 1))
    grad = np.dstack((np.cos(angles), np.sin(angles)))

    def tile(gy, gx):
        return grad[gy[0]:gy[1] or None, gx[0]:gx[1] or None] \
            .repeat(d[0], 0).repeat(d[1], 1)

    g00, g10 = tile((0, -1), (0, -1)), tile((1, 0), (0, -1))
    g01, g11 = tile((0, -1), (1, 0)), tile((1, 0), (1, 0))

    u, v = grid[:, :, 0], grid[:, :, 1]
    n00 = (np.dstack((u, v)) * g00).sum(2)
    n10 = (np.dstack((u - 1, v)) * g10).sum(2)
    n01 = (np.dstack((u, v - 1)) * g01).sum(2)
    n11 = (np.dstack((u - 1, v - 1)) * g11).sum(2)

    t = 6 * grid ** 5 - 15 * grid ** 4 + 10 * grid ** 3   # smootherstep
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
    return np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)


def fbm(shape, rng, octaves, res, persistence=0.5):
    """Fractal Brownian motion - octaves of perlin at doubling frequency."""
    total = np.zeros(shape)
    amp, norm, r = 1.0, 0.0, res
    for _ in range(octaves):
        total += amp * perlin(shape, (r, r), rng)
        norm += amp
        amp *= persistence
        r *= 2
    return total / norm


# --------------------------------------------------------------- region build
def union_find(pairs, ids):
    parent = {i: i for i in ids}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return {i: find(i) for i in ids}


def close_pairs(labels, min_gap):
    """Unordered region pairs whose nearest approach is under min_gap."""
    dist, (iy, ix) = ndimage.distance_transform_edt(labels == 0,
                                                    return_indices=True)
    near = labels[iy, ix]
    worst = {}
    for a, b, ga, gb in ((near[:, :-1], near[:, 1:], dist[:, :-1], dist[:, 1:]),
                         (near[:-1, :], near[1:, :], dist[:-1, :], dist[1:, :])):
        sel = (a != b) & (a > 0) & (b > 0)
        gap = ga[sel] + gb[sel]                # medial pixel -> ~half the gap each
        for ka, kb, g in zip(a[sel], b[sel], gap):
            key = (min(ka, kb), max(ka, kb))
            if g < worst.get(key, np.inf):
                worst[key] = g
    return [k for k, g in worst.items() if g < min_gap]


def regions(shape, rng, args):
    """Threshold an fBm field into clean, well-separated labelled regions."""
    field = fbm(shape, rng, args.octaves, args.res)
    mask = field > np.quantile(field, 1 - args.land)

    st = disk(args.smooth)
    mask = ndimage.binary_opening(mask, structure=st)
    mask = ndimage.binary_closing(mask, structure=st)
    mask = ndimage.binary_fill_holes(mask)

    # Shrink everything uniformly: this widens every inter-island gap by twice
    # the radius, which buys road clearance while keeping islands distinct.
    # Merging them instead would collapse most of the map into one region.
    mask = ndimage.binary_erosion(mask, structure=disk(args.separate))

    labels, _ = ndimage.label(mask)
    labels = drop_small(labels, args.min_area)

    # Whatever is still too close to fit a road between gets merged as a
    # fallback, so no road has to squeeze through a gap narrower than it is.
    for _ in range(6):
        pairs = close_pairs(labels, args.min_gap)
        if not pairs:
            break
        ids = [i for i in np.unique(labels) if i > 0]
        root = union_find(pairs, ids)
        lut = np.zeros(labels.max() + 1, dtype=labels.dtype)
        for i in ids:
            lut[i] = root[i]
        labels = lut[labels]

    return compact(labels)


def drop_small(labels, min_area):
    counts = np.bincount(labels.ravel())
    lut = np.arange(labels.max() + 1, dtype=labels.dtype)
    lut[counts[:len(lut)] < min_area] = 0
    lut[0] = 0
    return lut[labels]


def compact(labels):
    ids = [i for i in np.unique(labels) if i > 0]
    lut = np.zeros(labels.max() + 1, dtype=np.int32)
    for new, old in enumerate(ids, start=1):
        lut[old] = new
    return lut[labels]


# --------------------------------------------------------------------- render
def disk(r):
    ky, kx = np.mgrid[-r:r + 1, -r:r + 1]
    return (ky ** 2 + kx ** 2) <= r ** 2


def thick_edges(label, width):
    e = np.zeros(label.shape, dtype=bool)
    e[:, :-1] |= label[:, :-1] != label[:, 1:]
    e[:, 1:] |= label[:, 1:] != label[:, :-1]
    e[:-1, :] |= label[:-1, :] != label[1:, :]
    e[1:, :] |= label[1:, :] != label[:-1, :]
    return ndimage.binary_dilation(e, structure=disk((width - 2) // 2))


def build(seed, args):
    rng = np.random.default_rng(seed)
    shape = (args.size, args.size)
    labels = regions(shape, rng, args)
    n = int(labels.max())
    if n < 2:
        raise RuntimeError(f"seed {seed}: only {n} region(s) survived filtering")

    # generalized Voronoi of the regions -> organic roads
    _, (iy, ix) = ndimage.distance_transform_edt(labels == 0, return_indices=True)
    roads = thick_edges(labels[iy, ix], args.line)

    arr = np.full((*shape, 3), BG, dtype=np.uint8)
    # deal colours round-robin then shuffle, so no tile comes out near-monochrome
    palette = [COLORS[i % len(COLORS)] for i in range(n)]
    rng.shuffle(palette)
    for i, c in enumerate(palette, start=1):
        arr[labels == i] = c
    arr[roads] = ROAD

    img = Image.fromarray(arr)
    d = ImageDraw.Draw(img)

    pad = args.mark_r + args.line
    cand = np.argwhere(roads)
    cand = cand[(cand[:, 0] > pad) & (cand[:, 0] < args.size - pad) &
                (cand[:, 1] > pad) & (cand[:, 1] < args.size - pad)]
    chosen = []
    guard = 0
    while len(chosen) < args.marks and len(cand) and guard < 10000:
        guard += 1
        ry, rx = cand[rng.integers(len(cand))]
        if all(np.hypot(ry - cy, rx - cx) > args.mark_sep for cy, cx in chosen):
            chosen.append((ry, rx))
    for cy, cx in chosen:
        d.ellipse([cx - args.mark_r, cy - args.mark_r,
                   cx + args.mark_r, cy + args.mark_r],
                  fill=MARK_FILL, outline=MARK_RING, width=2)

    clear = ndimage.distance_transform_edt(~roads)[labels > 0].min()
    return img, n, clear, len(chosen)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=".", help="output directory")
    p.add_argument("--rows", type=int, default=2)
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--seed", type=int, default=200, help="seed of the first map")
    p.add_argument("--size", type=int, default=512, help="tile size in px")
    p.add_argument("--octaves", type=int, default=3)
    p.add_argument("--res", type=int, default=8, help="base lattice cells per side")
    p.add_argument("--land", type=float, default=0.28,
                   help="fraction of area above the threshold, before erosion; "
                        "much above ~0.4 the noise percolates into one mass")
    p.add_argument("--smooth", type=int, default=3, help="outline cleanup radius")
    p.add_argument("--separate", type=int, default=6,
                   help="erode regions by this to widen the gaps between them")
    p.add_argument("--min-area", type=int, default=600, help="drop regions below")
    p.add_argument("--min-gap", type=int, default=34,
                   help="merge regions closer than this, so roads always fit")
    p.add_argument("--line", type=int, default=10, help="road width in px")
    p.add_argument("--marks", type=int, default=2, help="yellow dots per map")
    p.add_argument("--mark-r", type=int, default=11)
    p.add_argument("--mark-sep", type=int, default=120)
    p.add_argument("--gutter", type=int, default=12)
    args = p.parse_args()

    step = args.res * 2 ** (args.octaves - 1)
    if args.size % step:
        p.error(f"--size must be a multiple of {step} for "
                f"res={args.res}, octaves={args.octaves}")

    tiles_dir = os.path.join(args.out, "perlin_maps")
    os.makedirs(tiles_dir, exist_ok=True)

    imgs = []
    for i in range(args.rows * args.cols):
        seed = args.seed + i
        img, n, clear, marks = build(seed, args)
        img.save(os.path.join(tiles_dir, f"perlin_{i + 1}.png"))
        imgs.append(img)
        print(f"map {i + 1} (seed {seed}): {n} regions | "
              f"min region-to-road gap {clear:.1f}px | {marks} marks")

    s, g = args.size, args.gutter
    W = args.cols * s + (args.cols + 1) * g
    H = args.rows * s + (args.rows + 1) * g
    grid = Image.new("RGB", (W, H), (255, 255, 255))
    for i, img in enumerate(imgs):
        r, c = divmod(i, args.cols)
        grid.paste(img, (g + c * (s + g), g + r * (s + g)))

    out = os.path.join(args.out, f"perlin_grid_{args.rows}x{args.cols}.png")
    grid.save(out)
    print(f"\ngrid: {W}x{H} -> {out}  (tiles in {tiles_dir})")


if __name__ == "__main__":
    main()
