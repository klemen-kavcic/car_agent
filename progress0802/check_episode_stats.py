"""Checks that Goals + Terminated + MaxStep sums to Episodes in a training .out log,
per summary line - i.e. that every episode-ending path is reporting a Custom/* stat.
Any nonzero 'Other' means some episode ended without a matching stat.

Usage: python check_episode_stats.py path/to/run.out
"""
import re
import sys


def _get(line, pattern, cast):
    m = re.search(pattern, line)
    return cast(m.group(1)) if m else 0


def main(path):
    mismatches = 0
    checked = 0
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "Episodes:" not in line:
                continue
            step = _get(line, r"Step: (\d+)", int)
            episodes = _get(line, r"Episodes: (\d+)", int)
            goals = _get(line, r"Goals: (\d+)", int)
            terminated = _get(line, r"Terminated: (\d+)", int)
            maxstep = _get(line, r"MaxStep: (\d+)", int)
            other = episodes - goals - terminated - maxstep
            checked += 1
            if other != 0:
                mismatches += 1
                print(f"Step {step:>10}: Episodes={episodes} Goals={goals} "
                      f"Terminated={terminated} MaxStep={maxstep} -> Other={other}")

    print(f"\nChecked {checked} summary lines, {mismatches} with unaccounted episodes.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_episode_stats.py path/to/run.out")
        sys.exit(1)
    main(sys.argv[1])
