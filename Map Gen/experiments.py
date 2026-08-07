"""Parameter sweeps for voronoi_highways: one contact sheet per knob.

Each sheet holds the seed fixed and varies a single parameter, so any visual
difference is attributable to that parameter alone. Measured statistics are
printed as a table and captioned under each tile.

    python experiments.py --out experiments
"""
import argparse
import os
from dataclasses import replace

import numpy as np

from PIL import Image, ImageDraw, ImageFont

from voronoi_highways import Params, generate

TILE = 300
PAD, CAP = 10, 46
BGC = (255, 255, 255)
FGC = (20, 20, 20)
SUBC = (110, 110, 110)


def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                      # older Pillow
        return ImageFont.load_default()


F_TITLE, F_CAP, F_SUB = font(21), font(17), font(14)


def sheet(title, cells, path):
    """cells: list of (image, caption, subcaption)."""
    n = len(cells)
    W = PAD + n * (TILE + PAD)
    H = PAD + 34 + TILE + CAP + PAD
    img = Image.new("RGB", (W, H), BGC)
    d = ImageDraw.Draw(img)
    d.text((PAD, PAD + 4), title, font=F_TITLE, fill=FGC)

    for i, (tile, cap, sub) in enumerate(cells):
        x = PAD + i * (TILE + PAD)
        y = PAD + 34
        img.paste(tile.resize((TILE, TILE), Image.LANCZOS), (x, y))
        d.rectangle([x, y, x + TILE - 1, y + TILE - 1], outline=(200, 200, 200))
        d.text((x, y + TILE + 7), cap, font=F_CAP, fill=FGC)
        d.text((x, y + TILE + 26), sub, font=F_SUB, fill=SUBC)

    img.save(path)
    return path


def run(name, title, values, mutate, caption, seed, base, outdir, rows, reps):
    """Sweep one parameter, averaging each setting over `reps` seeds.

    A single seed is far too noisy to read: changing any parameter reshuffles
    the ridges and with them the Voronoi topology, which moves the statistics
    more than the parameter itself does. The tile shown is the first seed; the
    numbers are the mean over all of them.
    """
    cells = []
    print(f"\n{title}   (mean +- sd over {reps} seeds)")
    print(f"  {'value':>12} | {'regions':>7} | {'gap px':>10} | "
          f"{'cover':>11} | {'wander px':>12}")
    print("  " + "-" * 64)
    for v in values:
        p = mutate(base, v)
        runs = [generate(seed + k, p) for k in range(reps)]
        gap = np.array([r.min_gap for r in runs])
        cov = np.array([r.cover for r in runs])
        wan = np.array([r.wander for r in runs])
        n = np.mean([r.n_regions for r in runs])

        cells.append((runs[0].image, caption(v),
                      f"gap {gap.mean():.0f}px  cover {cov.mean():.0%}  "
                      f"wander {wan.mean():.1f}px"))
        print(f"  {str(v):>12} | {n:>7.0f} | {gap.mean():>5.1f}+-{gap.std():<4.1f} | "
              f"{cov.mean():>6.0%}+-{cov.std():<4.0%} | "
              f"{wan.mean():>6.1f}+-{wan.std():<5.1f}")
        rows.append((name, str(v), n, gap.mean(), cov.mean(),
                     wan.mean(), wan.std()))
    path = sheet(title, cells, os.path.join(outdir, f"sweep_{name}.png"))
    print(f"  -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments")
    ap.add_argument("--seed", type=int, default=104)
    ap.add_argument("--reps", type=int, default=5,
                    help="seeds averaged per setting")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # fixed counts everywhere except the counts sweep, so tiles stay comparable
    base = Params(counts=(4, 4, 4))
    rows = []

    run("meander", "Road meander  (meander: 0 straight -> 2 winding)",
        [0.0, 0.5, 1.0, 1.5, 2.0],
        lambda b, v: replace(b, meander=v),
        lambda v: f"meander = {v}", a.seed, base, a.out, rows, a.reps)

    run("grow", "Region fill  (grow: spread from the ridge, px)",
        [0, 15, 30, 45, 60],
        lambda b, v: replace(b, grow=v),
        lambda v: f"grow = {v}", a.seed, base, a.out, rows, a.reps)

    run("buffer", "Road/region gap  (buffer, px)",
        [4, 10, 16, 24, 32],
        lambda b, v: replace(b, buffer=v),
        lambda v: f"buffer = {v}", a.seed, base, a.out, rows, a.reps)

    run("counts", "Region count per colour  (reds, greens, blues)",
        [(2, 2, 2), (3, 4, 2), (5, 5, 5), (7, 7, 7), (3, 8, 1)],
        lambda b, v: replace(b, counts=v),
        lambda v: f"R{v[0]} G{v[1]} B{v[2]}", a.seed, base, a.out, rows, a.reps)

    run("line", "Road width  (line, px)",
        [4, 10, 18, 28],
        lambda b, v: replace(b, line=v),
        lambda v: f"line = {v}px", a.seed, base, a.out, rows, a.reps)

    print(f"\n{len(rows)} runs across 5 sheets in {a.out}/")


if __name__ == "__main__":
    main()
