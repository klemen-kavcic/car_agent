"""Prune TensorBoard event files for clearly inferior experiment configurations.

The selection is deliberately configuration-based rather than seed-based: all seeds of a
configuration are kept together.  Within each experiment family, a configuration is retained if
it ranks in the top N by final-window validation goal rate for either deterministic or stochastic
evaluation.  Baselines, explicitly protected families, and configurations without enough data to
rank safely are always retained.

Dry run by default.  The script always writes audit CSVs before optionally deleting anything.

Example:
    python prune_tensorboard_events.py --root hpc
    python prune_tensorboard_events.py --root hpc --execute
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean


SEED_SUFFIX_RE = re.compile(r"_s\d+$", re.IGNORECASE)
RUN_PREFIX_RE = re.compile(r"^car_vgrid_\d+_", re.IGNORECASE)
BASELINE_TERMS = ("baseline", "default", "sensoroff")


def experiment_family(root: Path, run_dir: Path) -> str:
    """Use the path before results/ so obs137 and obs138 remain separate families."""
    relative = run_dir.relative_to(root)
    parts = relative.parts
    if "results" in parts:
        return str(Path(*parts[: parts.index("results")]))
    return str(relative.parent)


def condition_name(run_id: str) -> str:
    name = RUN_PREFIX_RE.sub("", run_id)
    return SEED_SUFFIX_RE.sub("", name)


def final_window_scores(eval_csv: Path, window: int) -> dict[str, tuple[float, float]]:
    """Return action_mode -> (goal rate, terminated rate) for the last N validation rows."""
    by_mode: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
    try:
        with eval_csv.open("r", newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if (row.get("maps") or "").strip().lower() != "val":
                    continue
                mode = (row.get("action_mode") or "").strip().lower()
                if mode not in ("deterministic", "stochastic"):
                    continue
                by_mode[mode].append(
                    (int(float(row["step"])), float(row["goal_rate"]), float(row["terminated_rate"]))
                )
    except (OSError, ValueError, KeyError, TypeError):
        return {}

    scores = {}
    for mode, rows in by_mode.items():
        rows.sort(key=lambda item: item[0])
        final_rows = rows[-window:]
        if final_rows:
            scores[mode] = (mean(row[1] for row in final_rows), mean(row[2] for row in final_rows))
    return scores


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, help="Directory to search recursively (for example hpc).")
    parser.add_argument("--top-n", type=int, default=5,
                        help="Keep the top N configurations in either action mode per family (default 5).")
    parser.add_argument("--final-window", type=int, default=5,
                        help="Number of final validation checkpoints averaged per seed (default 5).")
    parser.add_argument("--min-seeds", type=int, default=3,
                        help="Minimum scored seeds required before a configuration may be deleted (default 3).")
    parser.add_argument("--always-keep-family", action="append", default=["0907"],
                        help="Family path prefix to retain in full; repeatable (default 0907).")
    parser.add_argument("--manifest", default=None,
                        help="Per-event audit CSV (default ROOT/tensorboard_prune_manifest.csv).")
    parser.add_argument("--ranking", default=None,
                        help="Condition ranking CSV (default ROOT/tensorboard_condition_ranking.csv).")
    parser.add_argument("--execute", action="store_true",
                        help="Delete files marked DELETE. Without this flag, this is only a dry run.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        raise SystemExit(f"--root is not a directory: {root}")
    if args.top_n < 1 or args.final_window < 1 or args.min_seeds < 1:
        raise SystemExit("--top-n, --final-window and --min-seeds must all be positive.")

    manifest_path = Path(args.manifest) if args.manifest else root / "tensorboard_prune_manifest.csv"
    ranking_path = Path(args.ranking) if args.ranking else root / "tensorboard_condition_ranking.csv"

    event_records = []
    seed_scores: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for event_path in sorted(root.rglob("events.out.tfevents*")):
        if not event_path.is_file():
            continue
        run_dir = event_path.parent.parent
        family = experiment_family(root, run_dir)
        run_id = run_dir.name
        condition = condition_name(run_id)
        scores = final_window_scores(run_dir / "eval_results.csv", args.final_window)
        record = {
            "family": family,
            "run_dir": run_dir,
            "run_id": run_id,
            "condition": condition,
            "event_path": event_path,
            "size_bytes": event_path.stat().st_size,
            "scores": scores,
        }
        event_records.append(record)
        seed_scores[(family, condition)].append(record)

    condition_rows = []
    condition_stats = {}
    for (family, condition), records in seed_scores.items():
        stats = {"seed_count": len({record["run_id"] for record in records})}
        for mode in ("deterministic", "stochastic"):
            values = [record["scores"][mode] for record in records if mode in record["scores"]]
            stats[f"{mode}_seeds"] = len(values)
            stats[f"{mode}_goal_rate"] = mean(value[0] for value in values) if values else None
            stats[f"{mode}_terminated_rate"] = mean(value[1] for value in values) if values else None
        stats["rankable"] = all(stats[f"{mode}_seeds"] >= args.min_seeds
                                for mode in ("deterministic", "stochastic"))
        condition_stats[(family, condition)] = stats

    ranks: dict[tuple[str, str, str], int] = {}
    families = sorted({family for family, _ in condition_stats})
    for family in families:
        for mode in ("deterministic", "stochastic"):
            candidates = []
            for (candidate_family, condition), stats in condition_stats.items():
                if candidate_family != family or not stats["rankable"]:
                    continue
                candidates.append((
                    -stats[f"{mode}_goal_rate"],
                    stats[f"{mode}_terminated_rate"],
                    condition,
                ))
            for rank, (_, _, condition) in enumerate(sorted(candidates), start=1):
                ranks[(family, condition, mode)] = rank

    keep_conditions = set()
    reasons: dict[tuple[str, str], list[str]] = defaultdict(list)
    for key, stats in condition_stats.items():
        family, condition = key
        if any(family.lower().startswith(prefix.lower()) for prefix in args.always_keep_family):
            keep_conditions.add(key)
            reasons[key].append("protected_family")
        if any(term in condition.lower() for term in BASELINE_TERMS):
            keep_conditions.add(key)
            reasons[key].append("baseline")
        if not stats["rankable"]:
            keep_conditions.add(key)
            reasons[key].append("insufficient_data")
        for mode in ("deterministic", "stochastic"):
            rank = ranks.get((family, condition, mode))
            if rank is not None and rank <= args.top_n:
                keep_conditions.add(key)
                reasons[key].append(f"top_{args.top_n}_{mode}")

    for key in sorted(condition_stats):
        family, condition = key
        stats = condition_stats[key]
        decision = "KEEP" if key in keep_conditions else "DELETE"
        condition_rows.append({
            "family": family,
            "condition": condition,
            "decision": decision,
            "reason": ";".join(reasons[key]) if reasons[key] else "outside_top_set",
            "event_files": len(seed_scores[key]),
            "event_size_mib": sum(r["size_bytes"] for r in seed_scores[key]) / 2**20,
            "seed_count": stats["seed_count"],
            "deterministic_scored_seeds": stats["deterministic_seeds"],
            "deterministic_goal_rate": stats["deterministic_goal_rate"],
            "deterministic_terminated_rate": stats["deterministic_terminated_rate"],
            "deterministic_rank": ranks.get((family, condition, "deterministic")),
            "stochastic_scored_seeds": stats["stochastic_seeds"],
            "stochastic_goal_rate": stats["stochastic_goal_rate"],
            "stochastic_terminated_rate": stats["stochastic_terminated_rate"],
            "stochastic_rank": ranks.get((family, condition, "stochastic")),
        })

    manifest_rows = []
    delete_paths = []
    for record in event_records:
        key = (record["family"], record["condition"])
        decision = "KEEP" if key in keep_conditions else "DELETE"
        if decision == "DELETE":
            delete_paths.append(record["event_path"])
        manifest_rows.append({
            "family": record["family"],
            "run_id": record["run_id"],
            "condition": record["condition"],
            "decision": decision,
            "reason": ";".join(reasons[key]) if reasons[key] else "outside_top_set",
            "event_size_mib": record["size_bytes"] / 2**20,
            "event_path": str(record["event_path"].relative_to(root)),
        })

    write_csv(ranking_path, condition_rows, list(condition_rows[0]) if condition_rows else [])
    write_csv(manifest_path, manifest_rows, list(manifest_rows[0]) if manifest_rows else [])

    delete_bytes = sum(path.stat().st_size for path in delete_paths)
    keep_bytes = sum(record["size_bytes"] for record in event_records if record["event_path"] not in delete_paths)
    print(f"Scanned {len(event_records)} TensorBoard event file(s) across "
          f"{len(condition_stats)} configuration(s).")
    print(f"KEEP: {len(event_records) - len(delete_paths)} file(s), {keep_bytes / 2**30:.3f} GiB")
    print(f"DELETE: {len(delete_paths)} file(s), {delete_bytes / 2**30:.3f} GiB")
    print(f"Manifest: {manifest_path}")
    print(f"Ranking: {ranking_path}")

    if args.execute:
        for path in delete_paths:
            resolved = path.resolve()
            if root not in resolved.parents:
                raise RuntimeError(f"Refusing to delete path outside root: {resolved}")
        for path in delete_paths:
            path.unlink()
        print(f"Deleted {len(delete_paths)} TensorBoard event file(s).")
    elif delete_paths:
        print("Dry run only; nothing was deleted. Re-run with --execute after reviewing the CSVs.")


if __name__ == "__main__":
    main()
