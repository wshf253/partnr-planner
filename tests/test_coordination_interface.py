"""Pilot interface regressions without simulator dependencies."""

import ast
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


parser = load("habitat_llm/coordination/action_parser.py", "pilot_parser")
observations = load("habitat_llm/coordination/observations.py", "pilot_observations")


class InterfaceTests(unittest.TestCase):
    def test_reported_action_labels(self):
        for call in (
            "Pick[toy_food_1]",
            "Navigate[table_25]",
            "Place[toy_food_0, on, bed_16, None, None]",
            "Rearrange[toy_animal_4, on, table_25, None, None]",
            "Wait[]",
        ):
            for end in ("", "\nAssigned!"):
                result = parser.normalize_action_label(
                    "Thought: next\nAction: " + call + end
                )
                self.assertEqual(result, "Thought: next\n" + call)

    def test_bare_action_and_thought_preserved(self):
        text = "Thought: Action: Pick failed before\nPick[toy_food_1]"
        self.assertEqual(parser.normalize_action_label(text), text)

    def test_does_not_salvage_ambiguous_or_malformed_calls(self):
        for line in (
            "Action: Pick[cup",
            "Action: Pick[a] Place[b]",
            "Action: Pick[a]\nextra text",
            "Action: Pick[[a]]",
        ):
            self.assertEqual(parser.normalize_action_label(line), line)

    def test_real_upstream_parser(self):
        # Execute the real parser functions, excluding unrelated rendering/graph imports.
        tree = ast.parse((ROOT / "habitat_llm/llm/instruct/utils.py").read_text())
        names = {
            "zero_shot_action_parser",
            "actions_parser",
            "has_valid_square_brackets",
            "remove_non_alpha_left",
        }
        functions = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in names
        ]
        upstream = types.ModuleType("pilot_upstream")
        exec("from typing import Dict, Tuple", upstream.__dict__)
        exec(
            compile(
                ast.Module(body=functions, type_ignores=[]),
                "real_action_parser",
                "exec",
            ),
            upstream.__dict__,
        )
        name = "habitat_llm.llm.instruct.utils"
        with patch.dict(sys.modules, {name: upstream}):
            agents = [types.SimpleNamespace(uid=0)]
            for call, expected in [
                ("Pick[toy_food_1]", ("Pick", "toy_food_1", None)),
                ("Wait[]", ("Wait", None, None)),
            ]:
                actual = parser.coordination_action_parser(
                    agents, "Thought: next\nAction: " + call
                )
                self.assertEqual(actual[0], expected)
            # Unknown skills remain unknown; no nearest-name guessing.
            result = parser.coordination_action_parser(agents, "Action: Invented[cup]")
            self.assertEqual(result[0][0], "Invented")

    def test_public_identity_and_no_private_progress(self):
        scene = {
            "robot_position": [0, 0, 0],
            "human_position": [1, 0, 0],
            "objects": {
                str(i): {"held_by": holder, "at_goal": False}
                for i, holder in enumerate((0, 1, None))
            },
        }
        obs = observations.public_observation(scene, 17)
        self.assertEqual(obs["sim_step"], 17)
        for i, role in enumerate(("self", "partner", None)):
            self.assertEqual(obs["objects"][str(i)]["held_by_role"], role)
            self.assertNotIn("at_goal", obs["objects"][str(i)])
            self.assertNotIn("held_by_role", scene["objects"][str(i)])

    def test_prompt_identifies_self_partner_and_time(self):
        from test_coordination_prompt import planner

        obj = planner()
        obj.replan("Move cup", {}, {0: "fresh"})
        for phrase in (
            "agent 0 (self)",
            "partner, agent 1",
            "held_by=0 means YOU",
            "recent_robot_action_results contains YOUR",
            "current_observation is the latest",
            "Do not prefix the tool call",
        ):
            self.assertIn(phrase, obj.last_decision_prompt)


if __name__ == "__main__":
    unittest.main()
