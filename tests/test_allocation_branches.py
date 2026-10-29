import copy
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class FakeCoord:
    def reset(self):
        self.history = []

    def prepare_prompt(self, *args, **kwargs):
        return "prompt", {}

    def replan(self, *args):
        return {"llm_response": "LLM_RESUMED"}

    def _add_responses_to_prompt(self, responses):
        self.history.append(responses)
        return "result"


def setup(choice, directory):
    name = "habitat_llm.planner.coordination_robot_planner"
    stub = types.ModuleType(name)
    stub.CoordinationRobotPlanner = FakeCoord
    with patch.dict(sys.modules, {name: stub}):
        m = load("habitat_llm/planner/allocation_branch_planner.py", "branches_test")
    p = m.AllocationBranchPlanner()
    p.agents = [types.SimpleNamespace(uid=0)]
    p.reset()
    p.configure_branch(choice, directory)
    p.public_observation = {
        "sim_step": 424,
        "objects": {
            "kettle_0": {
                "held_by": None,
                "location": "counter_19",
                "target": "counter_19",
            },
            "plate_1": {
                "held_by": None,
                "location": "table_27",
                "target": "counter_19",
            },
            "milk_frother_cup_2": {
                "held_by": None,
                "location": "table_32",
                "target": "cabinet_37",
            },
        },
    }
    return p


class BranchTests(unittest.TestCase):
    def test_configure_then_run_instruction_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = setup("plate", tmp)
            p.reset()  # EvaluationRunner.run_instruction resets after configure.
            self.assertIn("Navigate[table_18]", p.replan("task", {}, {0: "world"})["llm_response"])
            p._add_responses_to_prompt({0: "Successful execution!"})
            p.reset()
            self.assertEqual(p.script_index, 0)
            self.assertEqual(p.branch_report["script_results"], [])
            p.configure_branch("cup", tmp)
            p.reset()
            self.assertEqual(p.branch_report["choice"], "cup")
            self.assertEqual(p.script[4], ("Navigate", "table_32"))

    def test_same_prefix_different_allocation_then_llm(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = []
            for choice in ["plate", "cup"]:
                p = setup(choice, tmp)
                actions = []
                for i in range(8):
                    actions.append(p.replan("task", {}, {0: "world"})["llm_response"])
                    p._add_responses_to_prompt({0: ""})
                    self.assertEqual(p.script_index, i)
                    p._add_responses_to_prompt({0: "Successful execution!"})
                self.assertTrue(p.branch_report["branch_completed"])
                self.assertEqual(
                    p.replan("task", {}, {0: "world"})["llm_response"], "LLM_RESUMED"
                )
                runs.append((actions, copy.deepcopy(p.branch_report)))
            self.assertEqual(runs[0][0][:4], runs[1][0][:4])
            self.assertNotEqual(runs[0][0][4:], runs[1][0][4:])
            self.assertEqual(
                runs[0][1]["branch_observation_sha256"],
                runs[1][1]["branch_observation_sha256"],
            )

    def test_prefix_failure_aborts_and_branch_failure_releases(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = setup("plate", tmp)
            p.replan("task", {}, {0: "world"})
            with self.assertRaisesRegex(RuntimeError, "prefix failed"):
                p._add_responses_to_prompt({0: "Failed"})
            p = setup("cup", tmp)
            p.script_index = 4
            p.replan("task", {}, {0: "world"})
            p._add_responses_to_prompt({0: "Failed"})
            self.assertTrue(p.branch_report["branch_failed"])
            self.assertFalse(p.branch_report["branch_completed"])
            self.assertEqual(
                p.replan("task", {}, {0: "world"})["llm_response"], "LLM_RESUMED"
            )

    def test_unavailable_alternative_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = setup("plate", tmp)
            p.script_index = 4
            p.public_observation["objects"]["plate_1"]["held_by"] = 1
            with self.assertRaisesRegex(RuntimeError, "no longer both available"):
                p.replan("task", {}, {0: "world"})

    def test_summary_does_not_compare_unmatched_or_failed_runs(self):
        m = load("scripts/summarize_allocation_branches.py", "branch_summary_test")
        a = dict(
            partner_id="p",
            episode_id="24318",
            seed=1,
            condition="plate",
            sim_steps=100,
            initial_observation_sha256="initial",
            branch_observation_sha256="branch",
            task_state_success=1,
            branch_completed=True,
            branch_failed=False,
        )
        b = dict(a, condition="cup", sim_steps=80)
        self.assertEqual(
            m.summarize([a, b])["comparisons"][0]["steps_saved_by_cup"], 20
        )
        for override in [
            dict(branch_observation_sha256="different"),
            dict(branch_completed=False),
            dict(error="failed"),
        ]:
            self.assertIsNone(
                m.summarize([a, dict(b, **override)])["comparisons"][0][
                    "steps_saved_by_cup"
                ]
            )
