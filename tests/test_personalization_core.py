"""Run without Habitat, model weights or a GPU: python -m unittest discover -s tests."""

import json
import tempfile
import unittest
from pathlib import Path

from habitat_llm.personalization.core import (
    FeedbackMemory,
    PROFILES,
    TakeoverProtocol,
    build_prompt,
    parse_decision,
    placement_pair,
)


def episode():
    return {
        "evaluation_propositions": [
            {
                "function_name": "is_on_top",
                "args": {"object_handles": [obj], "receptacle_handles": ["table"]},
            }
            for obj in ("stand", "phone")
        ]
    }


class ProtocolTests(unittest.TestCase):
    def pause(self):
        protocol = TakeoverProtocol(3)
        protocol.tick()
        protocol.skill_finished("Successful execution!")
        return protocol

    def test_act_assigns_robot_and_finishes(self):
        protocol = self.pause()
        protocol.choose("ACT")
        self.assertEqual(protocol.phase, "robot_remaining")
        protocol.tick()
        protocol.skill_finished("Successful execution!")
        self.assertEqual(protocol.result("accept_help")["preference_match"], 1)
        self.assertEqual(protocol.result("keep_ownership")["unwanted_takeover"], 1)

    def test_wait_resumes_on_simulator_clock(self):
        protocol = self.pause()
        protocol.choose("WAIT")
        for _ in range(2):
            protocol.tick()
            self.assertEqual(protocol.phase, "pause")
        protocol.tick()
        self.assertEqual(protocol.phase, "human_remaining")
        protocol.skill_finished("Successful execution!")
        self.assertEqual(protocol.result("keep_ownership")["preference_match"], 1)

    def test_failures_do_not_produce_feedback(self):
        protocol = self.pause()
        protocol.choose("ACT")
        protocol.skill_finished("Could not reach object")
        result = protocol.result("accept_help")
        self.assertEqual(result["phase"], "failed")
        self.assertIsNone(result["feedback"])

    def test_first_move_failure_is_not_wait(self):
        protocol = TakeoverProtocol(3)
        protocol.skill_finished("failed")
        self.assertIsNone(protocol.result("keep_ownership")["preference_match"])

    def test_timeout_does_not_generate_feedback(self):
        protocol = self.pause()
        protocol.choose("WAIT")
        self.assertIsNone(protocol.result("keep_ownership")["feedback"])

    def test_single_decision_only(self):
        protocol = self.pause()
        protocol.choose("WAIT")
        with self.assertRaises(ValueError):
            protocol.choose("ACT")

    def test_parser_rejects_invalid_instead_of_defaulting(self):
        self.assertEqual(parse_decision('{"decision":"ACT"}')["decision"], "ACT")
        for text in (
            "WAIT",
            '{"decision":"MAYBE"}',
            "[]",
            '{"decision":"ACT","reason":1}',
        ):
            with self.assertRaises(ValueError):
                parse_decision(text)

    def test_profile_does_not_change_physical_protocol(self):
        for choice in ("ACT", "WAIT"):
            protocol = self.pause()
            protocol.choose(choice)
            phase = protocol.phase
            for profile in PROFILES:
                protocol.result(profile)
                self.assertEqual(protocol.phase, phase)


class MemoryTests(unittest.TestCase):
    def test_persistence_and_user_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            one = FeedbackMemory(Path(tmp) / "u1.json")
            two = FeedbackMemory(Path(tmp) / "u2.json")
            one.record(1, "ACT", "Please wait next time")
            restored = FeedbackMemory(one.path)
            self.assertIn("Please wait next time", restored.context())
            self.assertNotIn("Please wait", two.context())
            self.assertEqual(
                set(json.loads(one.path.read_text())[0]),
                {"day", "decision", "feedback"},
            )

    def test_prompt_has_no_hidden_profile_or_future_action(self):
        prompt = build_prompt("Move objects", "stand", "phone", "table", "No history")
        for hidden in (
            "accept_help",
            "keep_ownership",
            "user_01",
            "pause_steps",
            "human_remaining",
        ):
            self.assertNotIn(hidden, prompt)
        self.assertIn("phone", prompt)


class DatasetTests(unittest.TestCase):
    def test_exact_two_goals(self):
        self.assertEqual(
            placement_pair(episode()), [("stand", "table"), ("phone", "table")]
        )

    def test_reject_alternative_targets(self):
        ep = episode()
        ep["evaluation_propositions"][0]["args"]["receptacle_handles"].append("shelf")
        with self.assertRaises(ValueError):
            placement_pair(ep)

    def test_reject_richer_tasks_and_constraints(self):
        ep = episode()
        ep["evaluation_constraints"] = [
            {"type": "TemporalConstraint", "args": {"dag_edges": [[0, 1]]}}
        ]
        with self.assertRaises(ValueError):
            placement_pair(ep)
        ep = episode()
        ep["evaluation_propositions"][0]["function_name"] = "is_inside"
        with self.assertRaises(ValueError):
            placement_pair(ep)

    def test_quantification_requires_both_objects(self):
        ep = {
            "evaluation_propositions": [
                {
                    "function_name": "is_on_top",
                    "args": {
                        "object_handles": ["phone", "stand"],
                        "receptacle_handles": ["table"],
                        "number": 2,
                    },
                }
            ]
        }
        self.assertEqual(len(placement_pair(ep)), 2)
        ep["evaluation_propositions"][0]["args"]["number"] = 1
        with self.assertRaises(ValueError):
            placement_pair(ep)


if __name__ == "__main__":
    unittest.main()
