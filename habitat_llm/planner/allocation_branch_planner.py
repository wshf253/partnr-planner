"""Replay a fixed prefix and one complete allocation, then resume the same LLM."""

import copy
import hashlib
import json

from habitat_llm.planner.coordination_robot_planner import CoordinationRobotPlanner

PREFIX = [
    ("Navigate", "table_18"),
    ("Pick", "kettle_0"),
    ("Navigate", "counter_19"),
    ("Place", "kettle_0, on, counter_19, None, None"),
]
BRANCHES = {
    "plate": [
        ("Navigate", "table_27"),
        ("Pick", "plate_1"),
        ("Navigate", "counter_19"),
        ("Place", "plate_1, on, counter_19, None, None"),
    ],
    "cup": [
        ("Navigate", "table_32"),
        ("Pick", "milk_frother_cup_2"),
        ("Navigate", "cabinet_37"),
        ("Place", "milk_frother_cup_2, on, cabinet_37, None, None"),
    ],
}


class AllocationBranchPlanner(CoordinationRobotPlanner):
    def reset(self):
        super().reset()
        self.branch_report = None
        self.script_index = 0
        self.script_pending = False
        self.script_released = False
        self.decision_source = "llm"

    def configure_branch(self, choice, directory):
        from pathlib import Path

        if choice not in BRANCHES:
            raise ValueError("Branch choice must be plate or cup")
        self.script = PREFIX + BRANCHES[choice]
        self.report_path = Path(directory) / "branch_audit.json"
        self.branch_report = {
            "choice": choice,
            "prefix_success": False,
            "branch_completed": False,
            "branch_failed": False,
            "branch_observation": None,
            "branch_observation_sha256": None,
            "script_results": [],
        }
        self._save_branch()

    def _save_branch(self):
        self.report_path.write_text(json.dumps(self.branch_report, indent=2))

    def replan(self, instruction, observations, world_graph):
        if self.branch_report is None:
            raise RuntimeError(
                "Branch planner must be configured for a controlled trial"
            )
        if self.script_released or self.script_index >= len(self.script):
            self.decision_source = "llm"
            return super().replan(instruction, observations, world_graph)
        if self.script_index == len(PREFIX):
            obs = copy.deepcopy(self.public_observation)
            if any(o["held_by"] == 0 for o in obs["objects"].values()):
                raise RuntimeError(
                    "Robot is still holding an object at branch boundary"
                )
            kettle = obs["objects"]["kettle_0"]
            if kettle["location"] != "counter_19" or kettle["held_by"] is not None:
                raise RuntimeError("Common prefix did not put kettle on counter")
            if any(
                obs["objects"][name]["held_by"] is not None
                or obs["objects"][name]["location"] == obs["objects"][name]["target"]
                for name in ("plate_1", "milk_frother_cup_2")
            ):
                raise RuntimeError("Branch alternatives are no longer both available")
            self.branch_report["prefix_success"] = True
            self.branch_report["branch_observation"] = obs
            self.branch_report["branch_observation_sha256"] = hashlib.sha256(
                json.dumps(obs, sort_keys=True).encode()
            ).hexdigest()
            self._save_branch()
        phase = "prefix" if self.script_index < len(PREFIX) else "allocation"
        self.decision_source = "forced_" + phase
        # Use the regular action parser/executor; do not call the LLM for forced choices.
        self.curr_prompt, self.params = self.prepare_prompt(
            instruction, world_graph[self.agents[0].uid], observations=observations
        )
        self.last_decision_prompt = self.curr_prompt
        self.prompt_budget_info = {"forced": True}
        skill, args = self.script[self.script_index]
        self.script_pending = True
        return {"llm_response": f"Thought: Controlled {phase} action.\n{skill}[{args}]"}

    def _add_responses_to_prompt(self, responses):
        result = super()._add_responses_to_prompt(responses)
        response = responses.get(self.agents[0].uid, "")
        if self.branch_report is not None and self.script_pending and response:
            self.script_pending = False
            self.branch_report["script_results"].append(
                {
                    "index": self.script_index,
                    "action": self.script[self.script_index],
                    "response": response,
                    "sim_step": self.public_observation["sim_step"],
                }
            )
            if response != "Successful execution!":
                self.branch_report["branch_failed"] = self.script_index >= len(PREFIX)
                self.script_released = True
                self._save_branch()
                if self.script_index < len(PREFIX):
                    raise RuntimeError("Controlled common prefix failed: " + response)
                # Resume the same planner with the actual failure in its action history.
                return result
            self.script_index += 1
            if self.script_index == len(self.script):
                self.branch_report["branch_completed"] = True
            self._save_branch()
        return result
