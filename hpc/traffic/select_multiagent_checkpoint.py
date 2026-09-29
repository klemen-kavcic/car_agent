"""Resolve the saved ONNX nearest a requested million-step training milestone."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

CHECKPOINT_NAME = re.compile(r"^CarAgent-(\d+)\.onnx$")
INTERVAL = 1_000_000
MAX_DEVIATION = 100_000


def checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    directory = run_dir / "CarAgent"
    found = []
    for path in directory.glob("CarAgent-*.onnx"):
        match = CHECKPOINT_NAME.fullmatch(path.name)
        if match and path.stat().st_size > 0:
            found.append((int(match.group(1)), path))
    if not found:
        raise ValueError(f"no nonempty CarAgent-<step>.onnx checkpoints in {directory}")
    return found


def nearest(found: list[tuple[int, Path]], milestone: int) -> tuple[int, Path]:
    target = milestone * INTERVAL
    step, path = min(found, key=lambda item: abs(item[0] - target))
    if abs(step - target) > MAX_DEVIATION:
        raise ValueError(f"milestone {milestone}M has no checkpoint within "
                         f"{MAX_DEVIATION:,} steps; nearest is {step:,}")
    return step, path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--milestone", type=int)
    group.add_argument("--validate-all", action="store_true")
    args = parser.parse_args()
    if args.max_steps < INTERVAL or args.max_steps % INTERVAL:
        parser.error("--max-steps must be a positive whole number of millions")
    found = checkpoints(args.run_dir)
    last = args.max_steps // INTERVAL
    if args.validate_all:
        selected = [nearest(found, milestone) for milestone in range(1, last + 1)]
        if len({path for _, path in selected}) != last:
            raise ValueError("two milestones resolved to the same checkpoint")
        print(f"Validated {last} distinct million-step ONNX checkpoints in {args.run_dir}")
    else:
        if args.milestone < 1 or args.milestone > last:
            parser.error(f"--milestone must be 1..{last}")
        print(nearest(found, args.milestone)[1].resolve())


if __name__ == "__main__":
    main()
