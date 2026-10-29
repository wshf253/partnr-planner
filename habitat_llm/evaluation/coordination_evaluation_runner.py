"""Run a fixed ReAct robot against a scripted partner; log observable traces."""

import copy
import json
from pathlib import Path

from habitat_llm.coordination.core import distance
from habitat_llm.coordination.observations import public_observation, snapshot
from habitat_llm.evaluation.decentralized_evaluation_runner import (
    DecentralizedEvaluationRunner,
)


def action_object(action):
    if action and action[0] in ("Rearrange", "Pick", "Navigate") and action[1]:
        return str(action[1]).split(",", 1)[0].strip()
    return None


class CoordinationEvaluationRunner(DecentralizedEvaluationRunner):
    def configure_trial(self, goals, profile, context, cfg, trial_dir):
        self.goals, self.cfg = goals, cfg
        self.trial_dir = Path(trial_dir)
        self.trial_dir.mkdir(parents=True, exist_ok=False)
        self.planner[0].partner_context = context
        self.planner[0].max_prompt_chars = int(cfg.max_prompt_chars)
        self.planner[1].configure(
            profile, goals, int(cfg.human_poll_steps), float(cfg.yield_radius)
        )
        self.step_count = 0
        self.scene = snapshot(self.env_interface, goals)
        self.initial_observation = public_observation(self.scene, 0)
        self.observed = [self.initial_observation]
        self.last_info = {}
        self.last_actions = {}
        self.success = False
        self.stop_reason = None
        self.metrics = {
            "same_object_target_steps": 0,
            "robot_idle_with_available_work_steps": 0,
            "robot_distance": 0.0,
            "human_distance": 0.0,
            "robot_attempts_held_by_human": 0,
            "robot_attempts_already_at_goal": 0,
            "robot_skill_failures": 0,
            "robot_decisions": 0,
            "partner_choice_points": 0,
            "partner_style_divergence_points": 0,
            "robot_finished_before_team": False,
        }
        self._previous_robot_decision_count = 0
        (self.trial_dir / "partner_context.txt").write_text(context)

    def update_agent_action_history(self, planner_info):
        # Base implementation assumes every agent follows LLMPlanner's logging
        # schema. Mixed scripted/LLM execution uses events.jsonl instead.
        return

    def on_environment_step(self, info):
        previous = self.scene
        self.step_count += 1
        self.scene = snapshot(
            self.env_interface, self.goals, previous["robot_position"]
        )
        self.success = bool(info.get("task_state_success", False))
        self.metrics["robot_distance"] += distance(
            previous["robot_position"], self.scene["robot_position"]
        )
        self.metrics["human_distance"] += distance(
            previous["human_position"], self.scene["human_position"]
        )
        robot_action = self.last_actions.get(0)
        human_action = self.last_actions.get(1)
        robot_object, human_object = action_object(robot_action), action_object(
            human_action
        )
        if robot_object and robot_object == human_object:
            self.metrics["same_object_target_steps"] += 1
        available = any(
            not obj["at_goal"] and obj["held_by"] is None
            for obj in self.scene["objects"].values()
        )
        if available and (not robot_action or robot_action[0] in ("Wait", "Done")):
            self.metrics["robot_idle_with_available_work_steps"] += 1

        def signature(scene):
            return [
                (k, v["held_by"], v["location"]) for k, v in scene["objects"].items()
            ]

        if (
            signature(previous) != signature(self.scene)
            or self.step_count % int(self.cfg.observation_interval) == 0
        ):
            self.observed.append(public_observation(self.scene, self.step_count))

    def _terminal_info(self):
        info = copy.deepcopy(self.last_info)
        info["replanned"] = {0: False, 1: False}
        info["replan_required"] = {0: False, 1: False}
        info["responses"] = {0: "", 1: ""}
        info["high_level_actions"] = self.last_actions
        info["prompts"] = {0: self.planner[0].curr_prompt, 1: "Scripted partner"}
        info["traces"] = {0: self.planner[0].trace, 1: ""}
        return info

    def get_low_level_actions(self, instruction, observations, world_graph):
        if self.success or self.step_count >= int(self.cfg.max_sim_steps):
            self.stop_reason = "task_success" if self.success else "sim_step_limit"
            return {}, self._terminal_info(), True
        robot, human = self.planner[0], self.planner[1]
        robot.public_observation = public_observation(self.scene, self.step_count)
        robot.recent_observations = self.observed[-3:]
        human.scene, human.sim_step = self.scene, self.step_count
        actions, info, _ = super().get_low_level_actions(
            instruction, observations, world_graph
        )
        # Private evaluator diagnostics never enter public_trace or partner history.
        audit = getattr(human, "last_choice_audit", None)
        if audit is not None:
            self.metrics["partner_choice_points"] += 1
            self.metrics["partner_style_divergence_points"] += int(
                audit["style_choice_diverges"]
            )
            with (self.trial_dir / "partner_choices.jsonl").open("a") as stream:
                stream.write(json.dumps(audit) + "\n")
            human.last_choice_audit = None
        # Robot Done means it stops contributing, not that the team task is done.
        if robot.is_done:
            self.metrics["robot_finished_before_team"] = True
            info.setdefault("high_level_actions", {})[0] = ("Done", "", None)
        info.setdefault("prompts", {})[
            1
        ] = "Scripted partner; profile is evaluator-private."
        info.setdefault("traces", {})[1] = ""
        self.last_actions = dict(info.get("high_level_actions", {}))
        self.last_info = info
        if robot.replanning_count > self._previous_robot_decision_count:
            self._previous_robot_decision_count = robot.replanning_count
            self.metrics["robot_decisions"] += 1
            action = self.last_actions.get(0)
            obj = self.scene["objects"].get(action_object(action))
            if obj is not None:
                self.metrics["robot_attempts_held_by_human"] += int(obj["held_by"] == 1)
                self.metrics["robot_attempts_already_at_goal"] += int(obj["at_goal"])
            record = {
                "sim_step": self.step_count,
                "decision_source": getattr(robot, "decision_source", "llm"),
                "observation": robot.public_observation,
                "prompt": robot.last_decision_prompt,
                "prompt_budget": dict(robot.prompt_budget_info),
                "thought": info.get("thought", {}).get(0),
                "action": action,
            }
            with (self.trial_dir / "robot_decisions.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
        response = info.get("responses", {}).get(0, "")
        if response and response != "Successful execution!":
            self.metrics["robot_skill_failures"] += 1
        # Full evaluator event log is NOT given back to the robot as history.
        if any(info.get("responses", {}).values()) or any(
            info.get("replanned", {}).values()
        ):
            with (self.trial_dir / "events.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "sim_step": self.step_count,
                            "actions": self.last_actions,
                            "responses": info.get("responses", {}),
                            "human_yield_count": human.policy.yield_count,
                        }
                    )
                    + "\n"
                )
        return actions, info, False

    def finish_trial(self, instruction, episode_id, info):
        final = public_observation(self.scene, self.step_count)
        if self.observed[-1] != final:
            self.observed.append(final)
        trace = {
            "episode_id": str(episode_id),
            "instruction": instruction,
            "initial_observation": self.initial_observation,
            "observations": self.observed,
            "task_success": bool(info.get("task_state_success", False)),
        }
        (self.trial_dir / "public_trace.json").write_text(json.dumps(trace, indent=2))
        metrics = dict(self.metrics)
        metrics.update(
            sim_steps=self.step_count,
            stop_reason=self.stop_reason or "planner_loop_limit",
            human_skill_failures=self.planner[1].skill_failures,
            human_yield_count=self.planner[1].policy.yield_count,
        )
        return trace, metrics
