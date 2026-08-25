"""Deletes numbered checkpoint .onnx files (CarAgent-<step>.onnx, under each run's CarAgent/
subfolder) that aren't close to a multiple of --keep-interval steps, to shrink already-downloaded
run folders without losing the ability to re-evaluate at a coarser resolution later. Never touches
the run-root CarAgent.onnx (the final model - used for eval_trajectories.py/the Unity sensor-weight
visualization) or anything outside CarAgent/ (eval_results.csv, trajectories.csv, configuration.yaml).

Standalone (no onnxruntime/mlagents_envs import) so it runs anywhere Python + argparse works,
including directly on HPC via the login node - no need for the apptainer/mlagents environment this
project's other scripts require.

DRY RUN BY DEFAULT - only prints what would be deleted and how much space it'd free. Pass --execute
to actually delete. This is irreversible: a deleted checkpoint can only come back by retraining
that seed from scratch, so review the dry-run output before re-running with --execute.

Usage:
    python prune_checkpoints.py --root "hpc/0824/results" --keep-interval 1000000
    python prune_checkpoints.py --root "hpc/0824/results" --keep-interval 1000000 --execute
"""
import argparse
import re
from pathlib import Path

CHECKPOINT_RE = re.compile(r"^CarAgent-(\d+)\.onnx$")


def find_run_dirs(root: Path):
    """A 'run dir' is any directory containing a CarAgent/ subfolder with numbered checkpoints -
    discovered this way (not by a fixed naming pattern) so it works across every hpc/* results
    layout without needing to know each one's run_id convention."""
    for carAgent_dir in root.rglob("CarAgent"):
        if carAgent_dir.is_dir() and any(CHECKPOINT_RE.match(p.name) for p in carAgent_dir.glob("*.onnx")):
            yield carAgent_dir.parent


def select_checkpoints_to_keep(checkpoints, keep_interval: int):
    """checkpoints: list of (step, path), any order. Keeps whichever checkpoint is closest to each
    multiple of keep_interval (0, keep_interval, 2*keep_interval, ...) up to the max step present,
    PLUS always the single highest-step checkpoint (the final one, regardless of how close it lands
    to a multiple - re-evaluating "the end of training" should never silently lose resolution)."""
    if not checkpoints:
        return set()
    steps = [s for s, _ in checkpoints]
    max_step = max(steps)
    keep_indices = set()
    target = 0
    while target <= max_step:
        idx = min(range(len(steps)), key=lambda i: abs(steps[i] - target))
        keep_indices.add(idx)
        target += keep_interval
    keep_indices.add(max(range(len(steps)), key=lambda i: steps[i]))
    return keep_indices


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="Directory to search recursively for run folders.")
    ap.add_argument("--keep-interval", type=int, default=1_000_000,
                     help="Keep roughly one checkpoint per this many steps (default 1,000,000).")
    ap.add_argument("--execute", action="store_true",
                     help="Actually delete. Without this, only prints what WOULD be deleted (dry run).")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"--root {root} is not a directory.")
        return

    total_freed = 0
    total_deleted = 0
    total_kept = 0
    run_count = 0
    empty_final_warnings = []

    for run_dir in sorted(find_run_dirs(root)):
        run_count += 1
        carAgent_dir = run_dir / "CarAgent"
        checkpoints = []
        for p in carAgent_dir.glob("*.onnx"):
            m = CHECKPOINT_RE.match(p.name)
            if m:
                checkpoints.append((int(m.group(1)), p))
        keep_indices = select_checkpoints_to_keep(checkpoints, args.keep_interval)

        # The final checkpoint (highest step) is NEVER deleted regardless of --keep-interval - see
        # select_checkpoints_to_keep. But "never delete it" doesn't mean it's actually usable: if
        # it's a 0-byte placeholder (not really downloaded - see the OneDrive-placeholder finding
        # this session), that's worth surfacing rather than silently trusting it's there.
        if checkpoints:
            final_step, final_onnx_path = max(checkpoints, key=lambda sp: sp[0])
            if final_onnx_path.stat().st_size == 0:
                empty_final_warnings.append(f"{run_dir.name}: final checkpoint "
                                             f"{final_onnx_path.name} is 0 bytes (not really downloaded)")
        # The run-root CarAgent.onnx (final-model copy used by eval_trajectories.py/the Unity
        # sensor-weight overlay) is separate from the numbered CarAgent/ checkpoints above and this
        # script never touches it either way - checked here purely for the same notify-if-empty
        # reason, since it's the single file most directly "the finished model."
        root_onnx = run_dir / "CarAgent.onnx"
        if root_onnx.is_file() and root_onnx.stat().st_size == 0:
            empty_final_warnings.append(f"{run_dir.name}: root CarAgent.onnx is 0 bytes (not really downloaded)")

        run_deleted = 0
        run_freed = 0
        for i, (step, onnx_path) in enumerate(checkpoints):
            if i in keep_indices:
                total_kept += 1
                continue
            # .pt (raw PyTorch weights, mlagents-learn's own resume format) is saved alongside
            # every .onnx export under the same step number - same "unused, safe to prune"
            # reasoning applies, and it's typically the larger of the two files.
            pt_path = onnx_path.with_suffix(".pt")
            paths_to_remove = [onnx_path] + ([pt_path] if pt_path.is_file() else [])
            size = sum(p.stat().st_size for p in paths_to_remove)
            run_deleted += 1
            run_freed += size
            if args.execute:
                for p in paths_to_remove:
                    p.unlink()
        total_deleted += run_deleted
        total_freed += run_freed
        if run_deleted:
            print(f"{run_dir.name}: {len(checkpoints)} checkpoints -> keeping "
                  f"{len(checkpoints) - run_deleted}, {'deleted' if args.execute else 'would delete'} "
                  f"{run_deleted} ({run_freed / 1e6:.1f} MB)")

    print()
    print(f"{run_count} run(s) scanned under {root}")
    print(f"{'Deleted' if args.execute else 'Would delete'} {total_deleted} checkpoint(s), "
          f"{'freed' if args.execute else 'would free'} {total_freed / 1e6:.1f} MB, "
          f"keeping {total_kept}")
    if not args.execute and total_deleted:
        print("\nDry run only - nothing was deleted. Re-run with --execute to actually delete.")

    if empty_final_warnings:
        print(f"\n/!\\ {len(empty_final_warnings)} finished-model file(s) are 0 bytes (never touched "
              f"by this script, but not actually usable as-is):")
        for w in empty_final_warnings:
            print(f"  - {w}")


if __name__ == "__main__":
    main()
