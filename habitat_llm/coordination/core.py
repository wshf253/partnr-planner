"""Simulator-independent partner policies, task validation and frozen context."""

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass

PROFILE_DESCRIPTIONS = {
    "batch": (
        "This partner first picks a nearby unfinished placement, then prefers to "
        "finish remaining placements to that same destination before switching "
        "destinations. They skip objects already carried by the robot."
    ),
    "nearest": (
        "At each completed skill, this partner chooses the nearest available "
        "unfinished object by straight-line distance, regardless of its destination. "
        "They skip objects already carried by the robot."
    ),
    "yielding": (
        "At skill boundaries this partner chooses a nearby unfinished object, but "
        "avoids an object when the robot is nearby, closer than the partner, and "
        "visibly approaching it. They wait if every available object is avoided. "
        "They do not interrupt a manipulation already in progress."
    ),
}
CONDITIONS = ("no_history", "history", "oracle_style")


def distance(a, b):
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


@dataclass(frozen=True)
class Goal:
    object: str
    target: str


def extract_goals(episode, min_objects=4, min_groups=2):
    """Accept exact placement tasks only; no alternatives or ordered constraints."""
    if episode.get("evaluation_proposition_dependencies"):
        raise ValueError("Dependent goals are unsupported")
    for constraint in episode.get("evaluation_constraints", []):
        kind, args = constraint["type"], constraint.get("args", {})
        if kind == "TemporalConstraint" and not args.get("dag_edges"):
            continue
        if kind == "TerminalSatisfactionConstraint":
            continue
        raise ValueError(f"Unsupported constraint: {kind}")
    goals = []
    for prop in episode["evaluation_propositions"]:
        args = prop["args"]
        objects = args.get("object_handles", [])
        targets = args.get("receptacle_handles", [])
        if (
            prop["function_name"] != "is_on_top"
            or len(targets) != 1
            or not objects
            or args.get("number", 1) != len(objects)
        ):
            raise ValueError("Need exact is_on_top placements with one target per goal")
        goals.extend(Goal(obj, targets[0]) for obj in objects)
    groups = Counter(goal.target for goal in goals)
    if len({goal.object for goal in goals}) != len(goals):
        raise ValueError("Repeated object goal")
    if len(goals) < min_objects or sum(n >= 2 for n in groups.values()) < min_groups:
        raise ValueError(
            "Need at least two destinations with at least two objects each"
        )
    return goals


def layout_fingerprint(episode):
    # Exclude instruction/ID/goal text so a relabeled identical physical scene is
    # not treated as a new layout. Preserve full object transforms, not just names.
    payload = {
        k: episode.get(k)
        for k in (
            "scene_id",
            "rigid_objs",
            "ao_states",
            "object_states",
            "start_position",
            "start_rotation",
            "name_to_receptacle",
        )
    }
    if not payload["scene_id"] or not payload["rigid_objs"]:
        raise ValueError("Cannot verify layout split without scene_id and rigid_objs")
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def choose_split(candidates, support_ids, test_ids, support_count, test_count):
    """Auto-select distinct layouts; reject explicitly overlapping splits."""
    by_id = {str(ep["episode_id"]): ep for ep in candidates}
    if len(by_id) != len(candidates):
        raise ValueError("Duplicate episode IDs")
    if support_count < 1 or test_count < 1:
        raise ValueError("support_count and test_count must be positive")
    used = set()

    def select(ids, count):
        explicit = ids is not None
        pool = [by_id[str(i)] for i in ids] if explicit else candidates
        selected = []
        for ep in pool:
            signature = layout_fingerprint(ep)
            if signature in used:
                if explicit:
                    raise ValueError(
                        "Support/test episodes must have distinct physical layouts"
                    )
                continue
            selected.append(ep)
            used.add(signature)
            if not explicit and len(selected) == count:
                break
        if not selected or (not explicit and len(selected) < count):
            raise ValueError(
                "Not enough compatible distinct layouts; inspect a larger dataset or pin IDs"
            )
        return selected

    # Reserve explicit test layouts before auto-selecting support episodes.
    if support_ids is None and test_ids is not None:
        test = select(test_ids, test_count)
        support = select(None, support_count)
    else:
        support = select(support_ids, support_count)
        test = select(test_ids, test_count)
    return support, test


class PartnerPolicy:
    def __init__(self, profile, yield_radius=2.0):
        if profile not in PROFILE_DESCRIPTIONS:
            raise ValueError(f"Unknown partner profile: {profile}")
        self.profile = profile
        self.yield_radius = yield_radius
        self.group = None
        self.yield_count = 0

    def select(self, goals, scene, blocked=()):
        # scene is an observable snapshot, never the robot's private goal/plan.
        pending = [
            g
            for g in goals
            if not scene["objects"][g.object]["at_goal"]
            and scene["objects"][g.object]["held_by"] != 0
            and g.object not in blocked
        ]
        if not pending:
            return None
        human, robot = scene["human_position"], scene["robot_position"]
        previous = scene.get("previous_robot_position", robot)
        if self.profile == "yielding":
            allowed = []
            for goal in pending:
                pos = scene["objects"][goal.object]["position"]
                rd, hd = distance(robot, pos), distance(human, pos)
                approaching = distance(previous, pos) - rd > 0.005
                if rd < self.yield_radius and rd < hd and approaching:
                    self.yield_count += 1
                else:
                    allowed.append(goal)
            pending = allowed
        if not pending:
            return None
        if self.profile == "batch":
            same_group = [g for g in pending if g.target == self.group]
            if same_group:
                pending = same_group
        selected = min(
            pending,
            key=lambda g: (
                distance(human, scene["objects"][g.object]["position"]),
                g.object,
            ),
        )
        self.group = selected.target
        return selected


def frozen_context(condition, support_traces, profile=None, max_chars=18000):
    if condition == "no_history":
        return "You have no previous experience with this partner."
    if condition == "oracle_style":
        return "Known partner behavior: " + PROFILE_DESCRIPTIONS[profile]
    if condition != "history" or not support_traces:
        raise ValueError("History condition needs prior observed interactions")
    # Only these public fields cross the evaluator -> robot boundary.
    allowed = {
        "episode_id",
        "instruction",
        "initial_observation",
        "observations",
        "task_success",
    }
    clean = [
        {key: trace[key] for key in allowed if key in trace} for trace in support_traces
    ]
    body = json.dumps(clean, ensure_ascii=False, sort_keys=True)
    if len(body) > max_chars:
        # Keep complete valid episodes with endpoints, downsample intermediate
        # observations deterministically. Never truncate JSON halfway through.
        while len(body) > max_chars and any(
            len(t.get("observations", [])) > 2 for t in clean
        ):
            for trace in clean:
                obs = trace.get("observations", [])
                if len(obs) > 2:
                    trace["observations"] = obs[:1] + obs[1:-1:2] + obs[-1:]
            body = json.dumps(clean, ensure_ascii=False, sort_keys=True)
        if len(body) > max_chars:
            raise ValueError(
                "History exceeds context budget even after downsampling; increase max_history_chars"
            )
    return (
        "Observed previous collaborations with this same partner on OTHER layouts. "
        "Infer useful behavior patterns; old object IDs do not identify current objects.\n"
        + body
    )


def partner_choice_audit(goals, scene, group, blocked=()):
    """Evaluator-only one-step alternatives, NOT optimal actions or rollouts.

    Both policies start with the actual partner's last selected destination.
    Fresh instances ensure probing never mutates the running partner policy.
    """
    choices = {}
    for profile in ("batch", "nearest"):
        probe = PartnerPolicy(profile)
        probe.group = group
        goal = probe.select(goals, scene, blocked)
        choices[profile] = goal.__dict__.copy() if goal else None
    return {
        "previous_selected_destination": group,
        "blocked_objects": sorted(blocked),
        "choices": choices,
        "style_choice_diverges": choices["batch"] != choices["nearest"],
        "candidate_distances": {
            goal.object: {
                "target": goal.target,
                "partner_distance": distance(
                    scene["human_position"], scene["objects"][goal.object]["position"]
                ),
                "robot_distance": distance(
                    scene["robot_position"], scene["objects"][goal.object]["position"]
                ),
            }
            for goal in goals
            if not scene["objects"][goal.object]["at_goal"]
            and scene["objects"][goal.object]["held_by"] != 0
            and goal.object not in blocked
        },
    }
