"""Runner lifecycle tests with fake low-level execution, no GPU required."""

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from habitat_llm.coordination.core import Goal


def scene():
    return {
        "human_position": [0, 0, 0],
        "robot_position": [0, 0, 0],
        "previous_robot_position": [0, 0, 0],
        "objects": {
            "cup": {
                "position": [1, 0, 0],
                "held_by": None,
                "location": "counter",
                "target": "table",
                "at_goal": False,
            }
        },
    }


class FakeBase:
    def get_low_level_actions(self, instruction, observations, world_graph):
        return (
            {1: "moving"},
            {
                "high_level_actions": {
                    0: ("Done", "", None),
                    1: ("Rearrange", "cup, on, table, None, None", None),
                },
                "responses": {1: ""},
                "prompts": {0: "robot prompt"},
                "traces": {0: "trace"},
                "replanned": {0: False},
            },
            True,
        )


def load_runner():
    name = "habitat_llm.evaluation.decentralized_evaluation_runner"
    stub = types.ModuleType(name)
    stub.DecentralizedEvaluationRunner = FakeBase
    path = (
        Path(__file__).resolve().parents[1]
        / "habitat_llm/evaluation/coordination_evaluation_runner.py"
    )
    spec = importlib.util.spec_from_file_location(
        "coordination_runner_under_test", path
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {name: stub}):
        spec.loader.exec_module(module)
    return module


class RunnerTests(unittest.TestCase):
    def setup_runner(self, tmp):
        module = load_runner()
        runner = module.CoordinationEvaluationRunner()
        robot = types.SimpleNamespace(
            is_done=True, replanning_count=0, curr_prompt="", trace=""
        )
        human = types.SimpleNamespace(
            configure=lambda *args: None,
            policy=types.SimpleNamespace(yield_count=0),
            skill_failures=0,
        )
        runner.planner = {0: robot, 1: human}
        runner.env_interface = object()
        cfg = types.SimpleNamespace(
            max_prompt_chars=2000,
            human_poll_steps=30,
            yield_radius=2,
            max_sim_steps=4,
            observation_interval=2,
        )
        with patch.object(module, "snapshot", return_value=scene()):
            runner.configure_trial(
                [Goal("cup", "table")], "nearest", "frozen", cfg, Path(tmp) / "trial"
            )
        return module, runner

    def test_robot_done_does_not_terminate_team(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, runner = self.setup_runner(tmp)
            actions, info, done = runner.get_low_level_actions("task", {}, {})
            self.assertFalse(done)
            self.assertIn(1, actions)
            self.assertTrue(runner.metrics["robot_finished_before_team"])
            self.assertEqual(set(info["prompts"]), {0, 1})

    def test_success_stops_team_and_does_not_replay_responses(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, runner = self.setup_runner(tmp)
            runner.success = True
            _, info, done = runner.get_low_level_actions("task", {}, {})
            self.assertTrue(done)
            self.assertEqual(runner.stop_reason, "task_success")
            self.assertEqual(info["responses"], {0: "", 1: ""})

    def test_deadline_not_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, runner = self.setup_runner(tmp)
            runner.step_count = 4
            self.assertTrue(runner.get_low_level_actions("task", {}, {})[2])
            self.assertFalse(runner.success)
            self.assertEqual(runner.stop_reason, "sim_step_limit")

    def test_step_metrics_use_actual_step_hook(self):
        with tempfile.TemporaryDirectory() as tmp:
            module, runner = self.setup_runner(tmp)
            runner.last_actions = {0: ("Wait", "", None)}
            with patch.object(module, "snapshot", return_value=scene()):
                runner.on_environment_step({"task_state_success": 0})
            self.assertEqual(runner.step_count, 1)
            self.assertEqual(runner.metrics["robot_idle_with_available_work_steps"], 1)
            self.assertNotIn("profile", runner.observed[0])

    def test_same_target_metric_and_observation_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            module, runner = self.setup_runner(tmp)
            runner.last_actions = {
                uid: ("Rearrange", "cup, on, table, None, None", None) for uid in (0, 1)
            }
            with patch.object(module, "snapshot", return_value=scene()):
                runner.on_environment_step({"task_state_success": 0})
            trace, metrics = runner.finish_trial("task", "1", {"task_state_success": 0})
            self.assertEqual(metrics["same_object_target_steps"], 1)
            self.assertNotIn("human_yield_count", trace)
            self.assertNotIn("actions", trace)
            self.assertNotIn("profile", trace)


if __name__ == "__main__":
    unittest.main()
