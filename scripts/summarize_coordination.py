#!/usr/bin/env python3
"""Summarize paired diagnostics, keeping failed/incomplete trials visible."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def summarize(rows):
    tests = [row for row in rows if row["phase"] == "test"]
    groups, pairs = defaultdict(list), defaultdict(dict)
    for row in tests:
        groups[(row["profile"], row["condition"])].append(row)
        key = (row["partner_id"], row["episode_id"], row["seed"])
        if row["condition"] in pairs[key]:
            raise ValueError(f"Duplicate trial: {key}, {row['condition']}")
        pairs[key][row["condition"]] = row
    summaries = []
    for (profile, condition), group in sorted(groups.items()):
        valid = [row for row in group if "error" not in row]
        item = {
            "profile": profile,
            "condition": condition,
            "trials": len(group),
            "execution_errors": len(group) - len(valid),
            "task_successes": sum(bool(r.get("task_state_success")) for r in valid),
        }
        for metric in (
            "sim_steps",
            "same_object_target_steps",
            "robot_idle_with_available_work_steps",
            "robot_distance",
            "human_distance",
            "robot_skill_failures",
            "human_skill_failures",
            "partner_choice_points",
            "partner_style_divergence_points",
        ):
            values = [row[metric] for row in valid if metric in row]
            item["mean_" + metric] = mean(values) if values else None
        summaries.append(item)
    comparisons = []
    for key, conditions in sorted(pairs.items()):
        base = conditions.get("no_history")
        if not base or "error" in base:
            continue
        for name in ("history", "oracle_style"):
            other = conditions.get(name)
            if not other or "error" in other:
                continue
            matched = base.get("initial_observation_sha256") == other.get(
                "initial_observation_sha256"
            )
            both_success = bool(base.get("task_state_success")) and bool(
                other.get("task_state_success")
            )
            comparisons.append(
                {
                    "partner_id": key[0],
                    "episode_id": key[1],
                    "seed": key[2],
                    "condition": name,
                    "matched_initial_state": matched,
                    "baseline_success": bool(base.get("task_state_success")),
                    "condition_success": bool(other.get("task_state_success")),
                    "steps_saved_when_both_succeed": (
                        base["sim_steps"] - other["sim_steps"]
                        if matched and both_success
                        else None
                    ),
                    "overlap_steps_reduced": (
                        base["same_object_target_steps"]
                        - other["same_object_target_steps"]
                        if matched
                        else None
                    ),
                }
            )
    return {
        "groups": summaries,
        "style_evidence_checks": [
            {
                "phase": row["phase"],
                "partner_id": row["partner_id"],
                "episode_id": row["episode_id"],
                "condition": row["condition"],
                "seed": row["seed"],
                "divergence_points": row.get("partner_style_divergence_points"),
                "status": (
                    "execution_error"
                    if "error" in row
                    else (
                        "not_instrumented"
                        if "partner_style_divergence_points" not in row
                        else (
                            "no_style_distinguishing_choice_observed"
                            if row["partner_style_divergence_points"] == 0
                            else "inspect_partner_choices_and_public_history"
                        )
                    )
                ),
            }
            for row in rows
        ],
        "paired_comparisons": comparisons,
        "note": "Overlap/idle counts are review signals, not ground-truth coordination errors. Compare steps only for paired successful trials.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in args.results.read_text().splitlines()
        if line.strip()
    ]
    print(json.dumps(summarize(rows), indent=2))
