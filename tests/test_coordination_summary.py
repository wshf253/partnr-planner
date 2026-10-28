"""Do not mistake fast failures or unmatched resets for coordination gains."""

import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "coord_summary",
    Path(__file__).resolve().parents[1] / "scripts/summarize_coordination.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def row(condition, success, steps, fingerprint="same"):
    return dict(
        phase="test",
        profile="batch",
        partner_id="p1",
        episode_id="2",
        seed=1,
        condition=condition,
        task_state_success=success,
        sim_steps=steps,
        initial_observation_sha256=fingerprint,
        same_object_target_steps=10,
    )


class SummaryTests(unittest.TestCase):
    def test_fast_failure_has_no_steps_saved(self):
        output = module.summarize(
            [row("no_history", True, 100), row("history", False, 10)]
        )
        self.assertIsNone(
            output["paired_comparisons"][0]["steps_saved_when_both_succeed"]
        )

    def test_unmatched_reset_has_no_steps_saved(self):
        output = module.summarize(
            [row("no_history", True, 100), row("oracle_style", True, 50, "different")]
        )
        self.assertIsNone(
            output["paired_comparisons"][0]["steps_saved_when_both_succeed"]
        )

    def test_paired_success_difference(self):
        output = module.summarize(
            [row("no_history", True, 100), row("oracle_style", True, 50)]
        )
        self.assertEqual(
            output["paired_comparisons"][0]["steps_saved_when_both_succeed"], 50
        )


if __name__ == "__main__":
    unittest.main()
