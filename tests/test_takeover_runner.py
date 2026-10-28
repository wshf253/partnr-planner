"""Exercise the real runner with fake skill execution, without Habitat assets."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class FakeDecentralizedRunner:
    def get_low_level_actions(self, instruction, observations, world_graph):
        actions, responses, high = {}, {}, {}
        for uid, planner in self.planner.items():
            high[uid] = planner.action
            actions[uid] = "zero_velocity" if planner.action[0] == "Wait" else "move"
            if planner.action[0] == "Rearrange":
                planner.calls += 1
                responses[uid] = planner.response if planner.calls % 2 == 0 else ""
            else:
                responses[uid] = ""
        return actions, {"responses": responses, "high_level_actions": high}, False


def load_runner():
    name = "habitat_llm.evaluation.decentralized_evaluation_runner"
    module = types.ModuleType(name)
    module.DecentralizedEvaluationRunner = FakeDecentralizedRunner
    path = (
        Path(__file__).resolve().parents[1]
        / "habitat_llm/evaluation/takeover_evaluation_runner.py"
    )
    spec = importlib.util.spec_from_file_location("takeover_runner_under_test", path)
    loaded = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {name: module}):
        spec.loader.exec_module(loaded)
    return loaded.TakeoverEvaluationRunner


class FakeLLM:
    def __init__(self, decision):
        self.decision = decision
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return '{"decision": "' + self.decision + '"}'


class RunnerTests(unittest.TestCase):
    def run_trial(self, decision, failure=False):
        runner = load_runner()()
        runner.planner = {
            uid: types.SimpleNamespace(
                calls=0, action=None, response="Successful execution!"
            )
            for uid in (0, 1)
        }
        if failure:
            runner.planner[0].response = "Navigation failed"
        llm = FakeLLM(decision)
        runner.configure_trial(
            ("stand", "phone", "table"), "No previous feedback", llm, 3
        )
        history = []
        for _ in range(30):
            actions, info, done = runner.get_low_level_actions("Move both", {}, {})
            history.append(info)
            self.assertLessEqual(
                sum(
                    action[0] == "Rearrange"
                    for action in info["high_level_actions"].values()
                ),
                1,
            )
            if done:
                return runner, llm, history
            if actions:
                runner.on_environment_step({})
        self.fail("Protocol did not terminate")

    def test_act_executes_robot_skill_and_calls_llm_once(self):
        runner, llm, history = self.run_trial("ACT")
        self.assertEqual(runner.protocol.phase, "done")
        self.assertEqual(len(llm.prompts), 1)
        self.assertTrue(
            any(h["high_level_actions"][0][0] == "Rearrange" for h in history)
        )
        self.assertEqual(history[0]["high_level_actions"][0][0], "Wait")

    def test_wait_resumes_human_and_never_moves_robot(self):
        runner, llm, history = self.run_trial("WAIT")
        self.assertEqual(runner.protocol.phase, "done")
        self.assertEqual(len(llm.prompts), 1)
        self.assertTrue(all(h["high_level_actions"][0][0] == "Wait" for h in history))
        pauses = [h for h in history if h["takeover"]["phase"] == "pause"]
        self.assertEqual(len(pauses), 3)

    def test_motor_failure_not_reported_as_completion(self):
        runner, _, _ = self.run_trial("ACT", failure=True)
        self.assertEqual(runner.protocol.phase, "failed")
        self.assertIsNone(runner.protocol.result("accept_help")["feedback"])

    def test_prompt_does_not_contain_profile_or_future(self):
        _, llm, _ = self.run_trial("ACT")
        self.assertNotIn("accept_help", llm.prompts[0])
        self.assertNotIn("human_remaining", llm.prompts[0])


if __name__ == "__main__":
    unittest.main()
