"""Environment value-of-information gate: partners x forced allocations x seeds.

Both agents are scripted, so no LLM is involved. The gate asks whether the best
robot allocation depends on the partner by more than seed-to-seed variation. Only
if it does is it worth evaluating whether an LLM planner can exploit partner
information on this task.
"""
import hashlib
import json
import re
from statistics import mean


def validate_gate(gate, partners):
    allocations = list(gate.allocations)
    ids = [a.id for a in allocations]
    if len(allocations) < 2 or len(set(ids)) != len(ids):
        raise ValueError("Need at least two uniquely named allocations")
    for allocation in allocations:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", allocation.id) or not allocation.targets:
            raise ValueError("Each allocation needs a plain id and nonempty targets")
    if len({tuple(sorted(a.targets)) for a in allocations}) != len(allocations):
        raise ValueError("Allocations must differ in their targets")
    if len(partners) < 2:
        raise ValueError("Need at least two partners to observe an allocation flip")
    for partner in partners:
        if (partner.profile == "preferred_destination") != bool(
            partner.get("preferred_targets")
        ):
            raise ValueError(
                f"{partner.id}: preferred_targets is required for, and only for, "
                "preferred_destination"
            )
    if int(gate.get("min_margin_steps", 0)) < 0:
        raise ValueError("min_margin_steps must be nonnegative")


def _valid(row):
    return (
        not row.get("error")
        and row.get("initial_state_matches") is True
        and row.get("task_state_success") == 1
        and row.get("stop_reason") == "task_success"
    )


def summarize(rows, min_margin_steps=0, partners=None, allocations=None, seeds=None):
    # Planned ids keep partial (in-progress) summaries from looking complete.
    partners = list(partners or dict.fromkeys(r["partner_id"] for r in rows))
    allocations = list(allocations or dict.fromkeys(r["allocation"] for r in rows))
    seeds = sorted(seeds or {r["seed"] for r in rows})
    cells = {}
    for row in rows:
        key = (row["partner_id"], row["allocation"])
        if row["seed"] in cells.setdefault(key, {}):
            raise ValueError("Duplicate gate trial")
        cells[key][row["seed"]] = row
    reasons = []
    per_partner = []
    means = {}
    for partner in partners:
        stats = {}
        for allocation in allocations:
            trials = cells.get((partner, allocation), {})
            steps = {s: t["sim_steps"] for s, t in trials.items() if _valid(t)}
            stats[allocation] = {
                "valid_seeds": len(steps),
                "planned_seeds": len(seeds),
                "sim_steps_by_seed": steps,
                "mean_sim_steps": mean(steps.values()) if steps else None,
                "min_sim_steps": min(steps.values()) if steps else None,
                "max_sim_steps": max(steps.values()) if steps else None,
                "seed_spread": max(steps.values()) - min(steps.values()) if steps else None,
                "robot_skill_failures": [
                    trials[s].get("robot_skill_failures") for s in sorted(trials)
                ],
                "human_skill_failures": [
                    trials[s].get("human_skill_failures") for s in sorted(trials)
                ],
                "same_object_target_steps": [
                    trials[s].get("same_object_target_steps") for s in sorted(trials)
                ],
            }
        complete = len(allocations) > 1 and all(
            stats[a]["valid_seeds"] == len(seeds) for a in allocations
        )
        best = separated = margin = None
        wins = {}
        if complete:
            ranked = sorted(allocations, key=lambda a: stats[a]["mean_sim_steps"])
            best, runner_up = ranked[0], ranked[1]
            margin = stats[runner_up]["mean_sim_steps"] - stats[best]["mean_sim_steps"]
            if margin == 0:
                best = None
                reasons.append(f"{partner}: allocations tie on mean sim_steps")
            # No seed of the best allocation may be slower than any seed of another.
            separated = best is not None and all(
                stats[best]["max_sim_steps"] + min_margin_steps
                < stats[a]["min_sim_steps"]
                for a in allocations
                if a != best
            )
            if best is not None and not separated:
                reasons.append(
                    f"{partner}: best allocation {best} overlaps others across seeds "
                    f"(margin {margin} steps, min_margin_steps {min_margin_steps})"
                )
            for seed in seeds:
                by_alloc = {a: stats[a]["sim_steps_by_seed"][seed] for a in allocations}
                fastest = min(by_alloc.values())
                winners = [a for a, v in by_alloc.items() if v == fastest]
                wins[seed] = winners[0] if len(winners) == 1 else "tie"
            means[partner] = {a: stats[a]["mean_sim_steps"] for a in allocations}
        else:
            reasons.append(
                f"{partner}: not every allocation/seed is a successful matched run"
            )
        per_partner.append(
            {
                "partner_id": partner,
                "complete": complete,
                "best_allocation": best,
                "margin_to_runner_up_steps": margin,
                "separated_across_seeds": separated,
                "faster_allocation_by_seed": wins,
                "allocations": stats,
            }
        )
    bests = [p["best_allocation"] for p in per_partner]
    flip = None not in bests and len(set(bests)) > 1
    if None not in bests and not flip:
        reasons.append(f"Every partner has the same best allocation: {bests[0]}")
    personalization = None
    if len(means) == len(partners) and partners:
        # Value of partner information: per-partner best vs. the best single
        # allocation used for every partner (equal weight per partner).
        fixed = {a: mean(means[p][a] for p in partners) for a in allocations}
        best_fixed = min(fixed, key=fixed.get)
        personalized = mean(min(means[p].values()) for p in partners)
        personalization = {
            "best_fixed_allocation": best_fixed,
            "best_fixed_mean_steps": fixed[best_fixed],
            "personalized_mean_steps": personalized,
            "gain_steps": fixed[best_fixed] - personalized,
        }
    passed = flip and all(p["separated_across_seeds"] for p in per_partner)
    return {
        "note": (
            "Both agents are scripted; this checks whether the task rewards partner "
            "information, not whether any planner exploits it. A forced allocation is "
            "a fixed robot policy (its allocation first, then nearest remaining), so a "
            "partner taking allocated objects is part of that policy's cost, not an "
            "invalid run. Errors, initial-state mismatches and unsuccessful runs are "
            "kept in trials but excluded from step comparisons. Seed spread measures "
            "simulator variation only, not LLM sampling variation."
        ),
        "gate_passed": bool(passed),
        "allocation_flips_with_partner": flip,
        "personalization": personalization,
        "failure_reasons": reasons,
        "min_margin_steps": min_margin_steps,
        "seeds": seeds,
        "partners": per_partner,
        "trials": rows,
    }


def run_gate(runner, test, config, out, trial):
    cfg, gate = config.coordination, config.allocation_gate
    planned = dict(
        min_margin_steps=int(gate.get("min_margin_steps", 0)),
        partners=[p.id for p in cfg.partners],
        allocations=[a.id for a in gate.allocations],
        seeds=[int(s) for s in cfg.seeds],
    )
    rows = []
    for seed in cfg.seeds:
        for partner in cfg.partners:
            preferred = tuple(partner.get("preferred_targets") or ())
            for allocation in gate.allocations:
                name = f"gate_{allocation.id}_{partner.id}_ep{test['episode_id']}_seed{seed}"
                directory = out / name
                row = dict(
                    episode_id=str(test["episode_id"]), seed=int(seed),
                    partner_id=partner.id, profile=partner.profile,
                    partner_preferred_targets=list(preferred),
                    allocation=allocation.id,
                    allocated_targets=list(allocation.targets),
                    log_directory=str(directory),
                )
                try:
                    trace, metrics = trial(
                        runner, None, test, partner.profile,
                        "Scripted robot; no partner information is used.",
                        cfg, int(seed), directory,
                        preferred_targets=preferred,
                        allocation=(allocation.id, tuple(allocation.targets)),
                    )
                    row.update(metrics)
                    row["initial_state_matches"] = json.loads(
                        (directory / "initial_state_check.json").read_text()
                    )["matches"]
                    row["public_trace_sha256"] = hashlib.sha256(
                        json.dumps(trace, sort_keys=True).encode()
                    ).hexdigest()
                    initial = trace["initial_observation"]["objects"]
                    final = trace["observations"][-1]["objects"]

                    def at_goal(state):
                        return state["held_by"] is None and state["location"] == state["target"]

                    # Allocated objects the partner placed (not already done at reset).
                    row["allocated_objects_placed_by_partner"] = [
                        obj for obj in row["allocated_objects"]
                        if at_goal(final[obj]) and not at_goal(initial[obj])
                        and obj not in row["robot_placed_objects"]
                    ]
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
                rows.append(row)
                with (out / "results.jsonl").open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                (out / "allocation_gate_summary.json").write_text(
                    json.dumps(summarize(rows, **planned), indent=2)
                )
                print(json.dumps(row), flush=True)
