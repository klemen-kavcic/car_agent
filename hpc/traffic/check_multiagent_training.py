"""Check that all 30 multi-car runs and all 1M-step ONNX files are ready."""

from __future__ import annotations

import argparse
from pathlib import Path

from select_multiagent_checkpoint import checkpoints, nearest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--train-array-id", required=True)
    args = parser.parse_args()
    if not args.train_array_id.isdecimal():
        parser.error("--train-array-id must be numeric")
    problems = []
    checked = 0
    for cars in (2, 5, 10):
        for method, last in (("finetune", 20), ("scratch", 40)):
            for seed in range(5):
                run_id = f"traffic{cars}_{args.train_array_id}_{method}_s{seed}"
                run = args.results_dir / run_id
                try:
                    for name in ("CarAgent.onnx", "configuration.yaml",
                                 "CarAgent/checkpoint.pt"):
                        path = run / name
                        if not path.is_file() or path.stat().st_size == 0:
                            raise ValueError(f"missing or empty {path}")
                    found = checkpoints(run)
                    selected = [nearest(found, milestone)[1] for milestone in range(1, last + 1)]
                    if len(set(selected)) != last:
                        raise ValueError("million-step milestones are not distinct")
                    checked += 1
                except ValueError as exc:
                    problems.append((run_id, str(exc)))
    print(f"Ready: {checked}/30 training runs")
    for run_id, problem in problems:
        print(f"INCOMPLETE {run_id}: {problem}")
    if problems:
        raise SystemExit(1)
    print("All 900 requested million-step checkpoints are present; evaluation can start.")


if __name__ == "__main__":
    main()
