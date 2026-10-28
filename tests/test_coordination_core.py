"""Partner-dependent choices and frozen evidence, without Habitat dependencies."""

import copy
import json
import unittest

from habitat_llm.coordination.core import (
    Goal,
    PartnerPolicy,
    partner_choice_audit,
    choose_split,
    extract_goals,
    frozen_context,
    layout_fingerprint,
)


def episode(identifier, offset=0):
    return {
        "episode_id": str(identifier),
        "instruction": "Move four objects",
        "scene_id": "house",
        "rigid_objs": [["asset", [[offset, 0, 0]]]],
        "evaluation_propositions": [
            {
                "function_name": "is_on_top",
                "args": {"object_handles": [obj], "receptacle_handles": [target]},
            }
            for obj, target in [
                ("a", "table"),
                ("b", "table"),
                ("c", "shelf"),
                ("d", "shelf"),
            ]
        ],
    }


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


GOALS = [Goal("a", "table"), Goal("b", "table"), Goal("c", "shelf"), Goal("d", "shelf")]


class PartnerTests(unittest.TestCase):
    def test_choice_audit_tracks_divergence_without_mutation(self):
        state = scene()
        state["objects"]["a"]["at_goal"] = True
        before = copy.deepcopy(state)
        audit = partner_choice_audit(GOALS, state, "table")
        self.assertEqual(audit["choices"]["batch"]["object"], "b")
        self.assertEqual(audit["choices"]["nearest"]["object"], "c")
        self.assertTrue(audit["style_choice_diverges"])
        self.assertEqual(state, before)
        # A cooldown/robot-held object cannot create a fake opportunity.
        state["objects"]["b"]["held_by"] = 0
        audit = partner_choice_audit(GOALS, state, "table", blocked=["c"])
        self.assertFalse(audit["style_choice_diverges"])
        self.assertEqual(set(audit["candidate_distances"]), {"d"})

    def test_choice_audit_does_not_invent_preference_at_start_or_end(self):
        state = scene()
        self.assertFalse(
            partner_choice_audit(GOALS, state, None)["style_choice_diverges"]
        )
        for item in state["objects"].values():
            item["at_goal"] = True
        audit = partner_choice_audit(GOALS, state, "table")
        self.assertEqual(audit["choices"], {"batch": None, "nearest": None})
        self.assertFalse(audit["style_choice_diverges"])

    def test_batch_and_nearest_diverge_after_first_placement(self):
        state = scene()
        batch, nearest = PartnerPolicy("batch"), PartnerPolicy("nearest")
        self.assertEqual(batch.select(GOALS, state).object, "a")
        self.assertEqual(nearest.select(GOALS, state).object, "a")
        state["objects"]["a"]["at_goal"] = True
        self.assertEqual(batch.select(GOALS, state).object, "b")
        self.assertEqual(nearest.select(GOALS, state).object, "c")

    def test_batch_changes_group_when_completed_or_carried(self):
        state = scene()
        policy = PartnerPolicy("batch")
        policy.select(GOALS, state)
        state["objects"]["a"]["at_goal"] = True
        state["objects"]["b"]["held_by"] = 0
        self.assertEqual(policy.select(GOALS, state).object, "c")

    def test_yields_to_approaching_robot_not_stationary_robot(self):
        state = scene()
        state["robot_position"] = [1.1, 0, 0]
        state["previous_robot_position"] = [1.2, 0, 0]
        policy = PartnerPolicy("yielding")
        self.assertNotEqual(policy.select(GOALS, state).object, "a")
        self.assertGreater(policy.yield_count, 0)
        state["previous_robot_position"] = state["robot_position"]
        self.assertEqual(policy.select(GOALS, state).object, "a")

    def test_failure_cooldown_leaves_other_work_available(self):
        state = scene()
        self.assertEqual(
            PartnerPolicy("nearest").select(GOALS, state, blocked=["a"]).object, "c"
        )

    def test_finished_or_robot_held_objects_never_selected(self):
        state = scene()
        for item in state["objects"].values():
            item["held_by"] = 0
        for profile in ("batch", "nearest", "yielding"):
            self.assertIsNone(PartnerPolicy(profile).select(GOALS, state))


class SplitTests(unittest.TestCase):
    def test_exact_task_filter(self):
        self.assertEqual(len(extract_goals(episode(1))), 4)
        ep = episode(1)
        ep["evaluation_propositions"][0]["args"]["receptacle_handles"].append("other")
        with self.assertRaises(ValueError):
            extract_goals(ep)

    def test_reject_one_group_and_ordered_tasks(self):
        ep = episode(1)
        for prop in ep["evaluation_propositions"]:
            prop["args"]["receptacle_handles"] = ["table"]
        with self.assertRaises(ValueError):
            extract_goals(ep)
        ep = episode(1)
        ep["evaluation_constraints"] = [
            {"type": "TemporalConstraint", "args": {"dag_edges": [[0, 1]]}}
        ]
        with self.assertRaises(ValueError):
            extract_goals(ep)

    def test_same_layout_new_id_cannot_leak_into_test(self):
        candidates = [episode(1), episode(2), episode(3, 1)]
        support, test = choose_split(candidates, None, None, 1, 1)
        self.assertEqual(support[0]["episode_id"], "1")
        self.assertEqual(test[0]["episode_id"], "3")
        with self.assertRaises(ValueError):
            choose_split(candidates, [1], [2], 1, 1)

    def test_reserves_explicit_test_before_auto_support(self):
        support, test = choose_split([episode(1), episode(2, 1)], None, [1], 1, 1)
        self.assertEqual(test[0]["episode_id"], "1")
        self.assertEqual(support[0]["episode_id"], "2")

    def test_not_enough_layouts_fails_instead_of_reusing(self):
        with self.assertRaises(ValueError):
            choose_split([episode(1)], None, None, 1, 1)

    def test_missing_layout_metadata_rejected(self):
        with self.assertRaises(ValueError):
            layout_fingerprint({"episode_id": "1"})


class HistoryTests(unittest.TestCase):
    def trace(self):
        return {
            "episode_id": "support_1",
            "instruction": "task",
            "profile": "SECRET",
            "human_future_goal": "SECRET",
            "initial_observation": {"objects": {}},
            "observations": [
                {"sim_step": i, "human_position": [i, 0, 0]} for i in range(40)
            ],
            "task_success": True,
        }

    def test_history_is_public_only_and_unchanged(self):
        trace = self.trace()
        before = copy.deepcopy(trace)
        context = frozen_context("history", [trace], "nearest", max_chars=1000)
        self.assertNotIn("SECRET", context)
        self.assertEqual(trace, before)
        body = json.loads(context.split("\n", 1)[1])
        self.assertEqual(body[0]["observations"][-1]["sim_step"], 39)

    def test_no_history_cannot_access_support_or_profile(self):
        a = frozen_context("no_history", [self.trace()], "nearest")
        b = frozen_context("no_history", [], "batch")
        self.assertEqual(a, b)

    def test_oracle_description_is_style_not_future_goal(self):
        context = frozen_context("oracle_style", [self.trace()], "batch")
        self.assertIn("destination", context)
        self.assertNotIn("support_1", context)
        self.assertNotIn("SECRET", context)

    def test_frozen_history_identical_across_test_trials(self):
        frozen = [self.trace()]
        self.assertEqual(
            frozen_context("history", frozen), frozen_context("history", frozen)
        )
        with self.assertRaises(ValueError):
            frozen_context("history", [])


if __name__ == "__main__":
    unittest.main()
