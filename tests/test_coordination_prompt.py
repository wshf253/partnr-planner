"""Bounded prompt regressions; exercise the real subclass without Habitat."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

OVERFLOW = (
    'vLLM HTTP 400: {"error":{"message":"You passed 15361 input tokens and '
    "requested 1024 output tokens. However, the model context length is only "
    '16384 tokens, resulting in a maximum input length of 15360 tokens.",'
    '"param":"input_tokens"}}'
)


class FakeBase:
    def reset(self):
        self.curr_prompt = ""
        self.trace = ""
        self.last_high_level_actions = {}

    def prepare_prompt(self, instruction, world_graph, **kwargs):
        return "TASK=" + instruction + "\nCURRENT_WORLD=" + world_graph, {}

    def replan(self, instruction, observations, world_graph):
        return {"llm_response": self.llm.generate(self.curr_prompt)}


class FakeLLM:
    def __init__(self, errors=()):
        self.errors = list(errors)
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        if self.errors:
            raise RuntimeError(self.errors.pop(0))
        return "Thought: move\nNavigate[cup]\nAssigned!"


def planner(errors=()):
    module_name = "habitat_llm.planner.zero_shot_react_planner"
    stub = types.ModuleType(module_name)
    stub.ZeroShotReactPlanner = FakeBase
    path = (
        Path(__file__).resolve().parents[1]
        / "habitat_llm/planner/coordination_robot_planner.py"
    )
    spec = importlib.util.spec_from_file_location("budget_planner_test", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {module_name: stub}):
        spec.loader.exec_module(module)
    obj = module.CoordinationRobotPlanner()
    obj.agents = [types.SimpleNamespace(uid=0)]
    obj.partner_context = "FROZEN_PARTNER_HISTORY"
    obj.max_prompt_chars = 100000
    obj.llm = FakeLLM(errors)
    obj.reset()
    obj.public_observation = {"human_position": [1, 2, 3]}
    obj.recent_observations = [{"step": i} for i in range(3)]
    return obj


class PromptTests(unittest.TestCase):
    def test_fifty_replans_do_not_accumulate_worlds_or_thoughts(self):
        obj = planner()
        for step in range(50):
            obj.curr_prompt += "OLD_THOUGHT_AND_SNAPSHOT" * 100
            obj.last_high_level_actions = {0: ("Navigate", "cup", None)}
            obj._add_responses_to_prompt({0: "Successful execution!"})
            obj.replan("Move cup", {}, {0: "fresh"})
        self.assertEqual(len(obj.action_results), 6)
        for prompt in obj.llm.prompts:
            self.assertNotIn("OLD_THOUGHT_AND_SNAPSHOT", prompt)
            self.assertEqual(prompt.count("FROZEN_PARTNER_HISTORY"), 1)
            self.assertEqual(prompt.count("CURRENT_WORLD="), 1)
        self.assertEqual(len(obj.llm.prompts[10]), len(obj.llm.prompts[-1]))

    def test_exact_reported_overflow_retries_with_less_context(self):
        obj = planner([OVERFLOW])
        for i in range(6):
            obj.action_results.append({"action": f"action{i}", "result": "failed"})
        obj.replan("Move cup", {}, {0: "fresh"})
        self.assertEqual(len(obj.llm.prompts), 2)
        self.assertLess(len(obj.llm.prompts[1]), len(obj.llm.prompts[0]))
        self.assertIn("action5", obj.llm.prompts[1])
        self.assertIn("FROZEN_PARTNER_HISTORY", obj.llm.prompts[1])
        self.assertEqual(obj.prompt_budget_info["context_overflow_retries"], 1)

    def test_other_errors_not_retried(self):
        for error in (
            "vLLM HTTP 500: server error",
            "LLM output reached max_tokens",
            "vLLM HTTP 400: invalid model",
        ):
            obj = planner([error])
            with self.assertRaises(RuntimeError):
                obj.replan("Move cup", {}, {0: "fresh"})
            self.assertEqual(len(obj.llm.prompts), 1)

    def test_essential_overflow_never_drops_partner_or_latest_failure(self):
        obj = planner([OVERFLOW] * 10)
        obj.action_results.append({"action": "Pick[cup]", "result": "LATEST_FAILURE"})
        with self.assertRaisesRegex(
            RuntimeError, "Frozen partner context was NOT changed"
        ):
            obj.replan("Move cup", {}, {0: "fresh"})
        self.assertEqual(len(obj.llm.prompts), 4)
        for prompt in obj.llm.prompts:
            self.assertIn("FROZEN_PARTNER_HISTORY", prompt)
            self.assertIn("LATEST_FAILURE", prompt)
        self.assertEqual(obj.partner_context, "FROZEN_PARTNER_HISTORY")

    def test_reset_does_not_leak_previous_episode_actions(self):
        obj = planner()
        obj.action_results.append({"action": "previous_episode", "result": "ok"})
        obj._turn_limit = 1
        obj.reset()
        obj.replan("New task", {}, {0: "new_world"})
        self.assertNotIn("previous_episode", obj.last_decision_prompt)
        self.assertIn("FROZEN_PARTNER_HISTORY", obj.last_decision_prompt)
        self.assertEqual(obj._turn_limit, 6)

    def test_character_guard_preserves_essential_content(self):
        obj = planner()
        obj.max_prompt_chars = 1
        with self.assertRaisesRegex(RuntimeError, "Essential coordination prompt"):
            obj.replan("Move cup", {}, {0: "fresh"})
        self.assertEqual(len(obj.llm.prompts), 0)


if __name__ == "__main__":
    unittest.main()
