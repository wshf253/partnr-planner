"""LLM-free robot that commits to a forced destination allocation.

Used by the allocation gate to test whether the ENVIRONMENT rewards partner
information before any LLM planner is evaluated. The robot reads only the public
observation; it never sees the partner profile or the partner's selected goal.
"""

import json
from pathlib import Path

from habitat_llm.coordination.core import allocation_choice
from habitat_llm.planner.takeover_skill_planner import TakeoverSkillPlanner


class ScriptedAllocationPlanner(TakeoverSkillPlanner):
    # Attributes the coordination runner sets or reads on the robot planner.
    partner_context = ""
    max_prompt_chars = 0
    public_observation = None
    recent_observations = ()

    def reset(self):
        super().reset()
        self.action = ("Wait", "", None)
        self.is_done = False
        self.curr_prompt = ""
        self.trace = ""
        self.last_decision_prompt = ""
        self.prompt_budget_info = {"scripted": True}
        self.decision_source = "scripted_wait"
        self.replanning_count = 0
        self.blocked_until = {}
        self.wait_started = 0
        self.skill_failures = 0
        self.allocation_report = None
        # run_instruction resets planners after the trial has been configured.
        # Keep configuration, but restart all execution/audit state.
        config = getattr(self, "_allocation_config", None)
        if config is not None:
            self.configure_allocation(*config)

    def configure_allocation(self, name, targets, goals, poll_steps, directory):
        unknown = set(targets) - {goal.target for goal in goals}
        if not targets or unknown:
            raise ValueError(
                "Allocation targets must be nonempty goal targets; unknown: "
                + ", ".join(sorted(unknown))
            )
        self._allocation_config = (name, targets, goals, poll_steps, directory)
        self.goals, self.targets = goals, tuple(targets)
        self.poll_steps = int(poll_steps)
        self.report_path = Path(directory) / "allocation_audit.json"
        self.allocation_report = {
            "allocation": name,
            "allocated_targets": list(self.targets),
            "allocated_objects": [g.object for g in goals if g.target in self.targets],
            "placed_by_robot": [],
            "skill_results": [],
        }

    def _save_allocation(self):
        self.report_path.write_text(json.dumps(self.allocation_report, indent=2))

    def get_next_action(self, instruction, observations, world_graph):
        if self.allocation_report is None:
            raise RuntimeError("Scripted allocation robot must be configured first")
        step = self.public_observation["sim_step"]
        if self.active is not None and self.active[0] == "Wait":
            if step - self.wait_started >= self.poll_steps:
                self.agents[0].reset()
                self.active = None
        if self.active is None:
            blocked = [obj for obj, until in self.blocked_until.items() if until > step]
            goal, phase = allocation_choice(
                self.goals, self.public_observation, self.targets, blocked
            )
            self.action = (
                ("Rearrange", f"{goal.object}, on, {goal.target}, None, None", None)
                if goal is not None
                else ("Wait", "", None)
            )
            self.decision_source = "scripted_" + phase
            self.wait_started = step
            # The runner logs one robot_decisions.jsonl row per counted decision.
            self.replanning_count += 1
        actions, info, _ = super().get_next_action(
            instruction, observations, world_graph
        )
        response = info["responses"].get(self.agents[0].uid, "")
        if response and self.action[0] == "Rearrange":
            obj = self.action[1].split(",", 1)[0].strip()
            self.allocation_report["skill_results"].append(
                {
                    "sim_step": step,
                    "object": obj,
                    "phase": self.decision_source,
                    "response": response,
                }
            )
            if response == "Successful execution!":
                self.allocation_report["placed_by_robot"].append(obj)
            else:
                # Same cooldown rule as the scripted partner, so neither side retries
                # an object the other agent just took on every step.
                self.skill_failures += 1
                self.blocked_until[obj] = step + self.poll_steps * 4
            self._save_allocation()
        return actions, info, False
