#!/usr/bin/env python3
"""Compare successful matched controlled allocations; keep all failures visible."""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def summarize(rows):
    groups = defaultdict(dict)
    for row in rows:
        key = (row["partner_id"], row["episode_id"], row["seed"])
        if row["condition"] in groups[key]:
            raise ValueError("Duplicate branch trial")
        groups[key][row["condition"]] = row
    output = []
    for key, pair in sorted(groups.items()):
        a, b = pair.get("plate", {}), pair.get("cup", {})
        matched = bool(a.get("branch_observation_sha256")) and a.get(
            "branch_observation_sha256"
        ) == b.get("branch_observation_sha256")
        initial_matched = bool(a.get("initial_observation_sha256")) and a.get(
            "initial_observation_sha256"
        ) == b.get("initial_observation_sha256")
        valid = (
            matched
            and initial_matched
            and all(
                r.get("task_state_success")
                and r.get("branch_completed")
                and not r.get("branch_failed")
                and not r.get("error")
                for r in (a, b)
            )
        )
        delta = a["sim_steps"] - b["sim_steps"] if valid else None
        output.append(
            {
                "partner_id": key[0],
                "episode_id": key[1],
                "seed": key[2],
                "matched_initial_observation": initial_matched,
                "matched_branch_observation": matched,
                "comparable_completed_allocations": bool(valid),
                "steps_saved_by_cup": delta,
                "faster_observed_branch": (
                    ("cup" if delta > 0 else "plate" if delta < 0 else "tie")
                    if delta is not None
                    else None
                ),
                "trials": pair,
            }
        )
    hashes = {r.get("branch_observation_sha256") for r in rows}
    return {
        "note": "Observed state hashes include rounded positions and step; not full simulator-state equivalence. Four controlled rollouts validate this allocation only, not optimal policies or personalization performance.",
        "all_trials_reached_same_observed_branch_state": len(hashes) == 1
        and None not in hashes
        and bool(rows),
        "comparisons": output,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            summarize(
                [
                    json.loads(l)
                    for l in args.results.read_text().splitlines()
                    if l.strip()
                ]
            ),
            indent=2,
        )
    )
