import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from habitat_llm.coordination.allocation_gate import summarize, validate_gate
from habitat_llm.coordination.core import (
    Goal,
    PartnerPolicy,
    allocation_choice,
    frozen_context,
)

ROOT = Path(__file__).resolve().parents[1]
GOALS = [Goal("a", "table"), Goal("b", "table"), Goal("c", "shelf"), Goal("d", "shelf")]


class Config(dict):
    __getattr__ = dict.__getitem__


def scene():
    return {
        "human_position": [0, 0, 0],
        "robot_position": [10, 0, 0],
        "previous_robot_position": [10, 0, 0],
        "objects": {
            name: {"position": [pos, 0, 0], "at_goal": False, "held_by": None}
            for name, pos in [("a", 1), ("b", 8), ("c", 2), ("d", 9)]
        },
    }


def observation():
    return {
        "sim_step": 0,
        "robot_position": [10, 0, 0],
        "human_position": [0, 0, 0],
        "objects": {
            name: {"position": [pos, 0, 0], "held_by": None, "location": "floor",
                   "target": target}
            for (name, pos), target in zip(
                [("a", 1), ("b", 8), ("c", 2), ("d", 9)],
                ["table", "table", "shelf", "shelf"],
            )
        },
    }


class PolicyTests(unittest.TestCase):
    def test_preferred_destination_partner(self):
        policy = PartnerPolicy("preferred_destination", preferred_targets=["shelf"])
        self.assertEqual(policy.select(GOALS, scene()).object, "c")
        state = scene()
        for name in ("c", "d"):
            state["objects"][name]["at_goal"] = True
        # Preferred work done: nearest of the rest.
        self.assertEqual(policy.select(GOALS, state).object, "a")
        policy = PartnerPolicy("preferred_destination", preferred_targets=["table"])
        self.assertEqual(policy.select(GOALS, scene()).object, "a")
        with self.assertRaises(ValueError):
            PartnerPolicy("preferred_destination")
        with self.assertRaises(ValueError):
            PartnerPolicy("nearest", preferred_targets=["shelf"])
        self.assertIn(
            "Preferred destination(s): shelf.",
            frozen_context("oracle_style", [], "preferred_destination",
                           preferred_targets=["shelf"]),
        )

    def test_allocation_choice_uses_public_state_only(self):
        obs = observation()
        self.assertEqual(allocation_choice(GOALS, obs, ["table"]), (GOALS[1], "allocation"))
        self.assertEqual(allocation_choice(GOALS, obs, ["shelf"]), (GOALS[3], "allocation"))
        # Partner-held and already placed objects are skipped; blocked too.
        obs["objects"]["b"]["held_by"] = 1
        obs["objects"]["a"]["location"] = "table"
        self.assertEqual(
            allocation_choice(GOALS, obs, ["table"]), (GOALS[3], "fallback")
        )
        self.assertEqual(
            allocation_choice(GOALS, obs, ["table"], blocked=["d"]), (GOALS[2], "fallback")
        )
        for name in ("c", "d"):
            obs["objects"][name]["location"] = "shelf"
        self.assertEqual(allocation_choice(GOALS, obs, ["table"]), (None, "wait"))


class FakeTakeover:
    def __init__(self):
        self.responses = []

    def reset(self):
        self.active = None

    def get_next_action(self, instruction, observations, world_graph):
        if self.active != self.action:
            self.active = self.action
        response = self.responses.pop(0) if self.responses else ""
        if response:
            self.active = None
        return {}, {"high_level_actions": {0: self.action}, "responses": {0: response}}, False


def load_planner():
    name = "habitat_llm.planner.takeover_skill_planner"
    stub = types.ModuleType(name)
    stub.TakeoverSkillPlanner = FakeTakeover
    spec = importlib.util.spec_from_file_location(
        "scripted_allocation_test", ROOT / "habitat_llm/planner/scripted_allocation_planner.py"
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {name: stub}):
        spec.loader.exec_module(module)
    return module


class PlannerTests(unittest.TestCase):
    def test_commits_to_allocation_and_survives_runner_reset(self):
        module = load_planner()
        with tempfile.TemporaryDirectory() as tmp:
            p = module.ScriptedAllocationPlanner()
            p.agents = [types.SimpleNamespace(uid=0, reset=lambda: None)]
            p.reset()
            with self.assertRaises(RuntimeError):
                p.get_next_action("task", {}, {})
            with self.assertRaises(ValueError):
                p.configure_allocation("bad", ["sofa"], GOALS, 30, tmp)
            p.configure_allocation("robot_table", ["table"], GOALS, 30, tmp)
            p.reset()  # run_instruction resets after configuration.
            p.public_observation = observation()
            p.responses = ["", "Successful execution!"]
            p.get_next_action("task", {}, {})
            self.assertEqual(p.action[1], "b, on, table, None, None")
            self.assertEqual(p.decision_source, "scripted_allocation")
            p.get_next_action("task", {}, {})
            self.assertEqual(p.replanning_count, 1)
            p.public_observation["objects"]["b"]["location"] = "table"
            p.public_observation["objects"]["a"]["held_by"] = 1
            p.responses = ["Failed to pick"]
            p.get_next_action("task", {}, {})
            self.assertEqual(p.action[1], "d, on, shelf, None, None")
            self.assertEqual(p.skill_failures, 1)
            # The failed object is on cooldown, so the next choice differs.
            p.get_next_action("task", {}, {})
            self.assertEqual(p.action[1], "c, on, shelf, None, None")
            audit = json.loads((Path(tmp) / "allocation_audit.json").read_text())
            self.assertEqual(audit["placed_by_robot"], ["b"])
            self.assertEqual(audit["allocated_objects"], ["a", "b"])
            self.assertEqual(len(audit["skill_results"]), 2)
            p.reset()
            self.assertEqual(p.allocation_report["placed_by_robot"], [])
            self.assertEqual(p.targets, ("table",))


def row(partner, allocation, seed, steps, **overrides):
    return dict(partner_id=partner, allocation=allocation, seed=seed, sim_steps=steps,
                initial_state_matches=True, task_state_success=1,
                stop_reason="task_success", **overrides)


def grid(steps):
    return [row(p, a, s, v) for (p, a), values in steps.items()
            for s, v in zip((1, 2, 3), values)]


class SummaryTests(unittest.TestCase):
    def test_flip_passes_and_reports_personalization_gain(self):
        result = summarize(grid({
            ("pA", "rA"): [1000, 1010, 1005], ("pA", "rB"): [900, 905, 910],
            ("pB", "rA"): [880, 890, 885], ("pB", "rB"): [1000, 990, 995],
        }))
        self.assertTrue(result["gate_passed"])
        self.assertEqual([p["best_allocation"] for p in result["partners"]], ["rB", "rA"])
        self.assertEqual(result["personalization"]["best_fixed_allocation"], "rA")
        self.assertEqual(result["personalization"]["gain_steps"], 50)
        self.assertEqual(result["partners"][0]["faster_allocation_by_seed"],
                         {1: "rB", 2: "rB", 3: "rB"})

    def test_same_best_overlap_and_invalid_runs_fail(self):
        same = summarize(grid({
            ("pA", "rA"): [900] * 3, ("pA", "rB"): [1000] * 3,
            ("pB", "rA"): [900] * 3, ("pB", "rB"): [1000] * 3,
        }))
        self.assertFalse(same["gate_passed"])
        self.assertEqual(same["personalization"]["gain_steps"], 0)
        overlap = summarize(grid({
            ("pA", "rA"): [1000, 1000, 1000], ("pA", "rB"): [900, 1010, 905],
            ("pB", "rA"): [880, 890, 885], ("pB", "rB"): [1000, 990, 995],
        }))
        self.assertTrue(overlap["allocation_flips_with_partner"])
        self.assertFalse(overlap["gate_passed"])
        margin = summarize(grid({
            ("pA", "rA"): [1000] * 3, ("pA", "rB"): [990] * 3,
            ("pB", "rA"): [900] * 3, ("pB", "rB"): [1000] * 3,
        }), min_margin_steps=20)
        self.assertFalse(margin["gate_passed"])
        rows = grid({
            ("pA", "rA"): [1000] * 3, ("pA", "rB"): [900] * 3,
            ("pB", "rA"): [900] * 3, ("pB", "rB"): [1000] * 3,
        })
        self.assertTrue(summarize(rows)["gate_passed"])
        for override in [dict(error="x"), dict(initial_state_matches=False),
                         dict(task_state_success=0), dict(stop_reason="sim_step_limit")]:
            broken = rows[:-1] + [dict(rows[-1], **override)]
            result = summarize(broken)
            self.assertFalse(result["gate_passed"])
            self.assertIsNone(result["personalization"])
        with self.assertRaises(ValueError):
            summarize(rows + rows[:1])

    def test_partial_progress_is_incomplete_not_crash(self):
        result = summarize([row("pA", "rA", 1, 900)], partners=["pA", "pB"],
                           allocations=["rA", "rB"], seeds=[1, 2, 3])
        self.assertFalse(result["gate_passed"])
        self.assertFalse(result["partners"][0]["complete"])

    def test_validate_gate(self):
        partners = [Config(id="pA", profile="preferred_destination", preferred_targets=["t"]),
                    Config(id="pB", profile="preferred_destination", preferred_targets=["s"])]
        good = Config(allocations=[Config(id="rA", targets=["t"]),
                                   Config(id="rB", targets=["s"])], min_margin_steps=0)
        validate_gate(good, partners)
        bad = [
            Config(good, allocations=good.allocations[:1]),
            Config(good, allocations=[good.allocations[0], Config(id="rB", targets=["t"])]),
            Config(good, allocations=[good.allocations[0], Config(id="r B", targets=["s"])]),
            Config(good, min_margin_steps=-1),
        ]
        for gate in bad:
            with self.assertRaises(ValueError):
                validate_gate(gate, partners)
        with self.assertRaises(ValueError):
            validate_gate(good, partners[:1])
        with self.assertRaises(ValueError):
            validate_gate(good, [partners[0], Config(id="pB", profile="nearest",
                                                     preferred_targets=["s"])])
        with self.assertRaises(ValueError):
            validate_gate(good, [partners[0], Config(id="pB", profile="preferred_destination")])


if __name__ == "__main__":
    unittest.main()
