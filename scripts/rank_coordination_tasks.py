#!/usr/bin/env python3
"""Rank observed diagnostic signals, never claim partner-dependent optimality."""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def read_rows(path):
    return (
        [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if path.exists()
        else []
    )


def rank(directory):
    groups = defaultdict(list)
    for row in read_rows(directory / "results.jsonl"):
        if row["phase"] == "test":
            groups[(row["episode_id"], row["seed"])].append(row)
    output = []
    for (episode, seed), rows in groups.items():
        item = {
            "episode_id": episode,
            "seed": seed,
            "trials": [],
            "divergence_points": 0,
            "decision_opportunities": 0,
            "execution_errors": 0,
        }
        for row in rows:
            name = f"test_{row['condition']}_{row['partner_id']}_ep{episode}_seed{seed}"
            folder = directory / name
            decisions = read_rows(folder / "robot_decisions.jsonl")
            choices = read_rows(folder / "partner_choices.jsonl")
            opportunities = []
            for d in decisions:
                pending = [
                    o
                    for o in d["observation"]["objects"].values()
                    if o["held_by"] is None and o["location"] != o["target"]
                ]
                if len(pending) >= 2:
                    opportunities.append(d["sim_step"])
            divergences = []
            for choice in choices:
                if not choice["style_choice_diverges"]:
                    continue
                future = [step for step in opportunities if step >= choice["sim_step"]]
                divergences.append(
                    {
                        "sim_step": choice["sim_step"],
                        "choices": choice["choices"],
                        "next_robot_choice_delay": (
                            min(future) - choice["sim_step"] if future else None
                        ),
                    }
                )
            item["divergence_points"] += len(divergences)
            item["decision_opportunities"] += len(opportunities)
            item["execution_errors"] += int("error" in row)
            item["trials"].append(
                {
                    "profile": row["profile"],
                    "condition": row["condition"],
                    "success": row.get("task_state_success"),
                    "stop_reason": row.get("stop_reason"),
                    "error": row.get("error"),
                    "sim_steps": row.get("sim_steps"),
                    "overlap_steps": row.get("same_object_target_steps"),
                    "robot_failures": row.get("robot_skill_failures"),
                    "partner_failures": row.get("human_skill_failures"),
                    "robot_decision_steps_with_alternatives": opportunities,
                    "divergences": divergences,
                    "log_directory": str(folder),
                }
            )
        profiles = {r["profile"] for r in rows if "error" not in r}
        item["both_profiles_observed"] = {"batch", "nearest"} <= profiles
        item["status"] = (
            "incomplete_or_error"
            if not item["both_profiles_observed"] or item["execution_errors"]
            else (
                "inspect_candidate"
                if item["divergence_points"] and item["decision_opportunities"]
                else "weak_observed_signal"
            )
        )
        output.append(item)
    return sorted(
        output,
        key=lambda x: (
            x["status"] != "inspect_candidate",
            -x["divergence_points"],
            -x["decision_opportunities"],
            x["episode_id"],
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "directory",
        type=Path,
        help="Folder containing results.jsonl and trial directories",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            {
                "note": "Heuristic shortlist only. Counts depend on rollout length. A later robot choice may occur after the opportunity has disappeared. Errors and truncated runs remain visible. No optimal allocation is established.",
                "candidates": rank(args.directory),
            },
            indent=2,
        )
    )
