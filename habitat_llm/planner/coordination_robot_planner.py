"""ReAct with frozen partner context and a bounded current-episode prompt."""

import json
from collections import deque

from habitat_llm.planner.zero_shot_react_planner import ZeroShotReactPlanner


def is_context_overflow(error):
    """Only retry the explicit local vLLM input-context rejection."""
    message = str(error)
    return (
        "vLLM HTTP 400:" in message
        and "input_tokens" in message
        and ("context length" in message or "maximum input length" in message)
    )


class CoordinationRobotPlanner(ZeroShotReactPlanner):
    partner_context = "No previous experience."
    public_observation = None
    recent_observations = ()
    last_decision_prompt = ""

    def reset(self):
        super().reset()
        self.action_results = deque(maxlen=6)
        self._turn_limit = 6
        self._observation_limit = 3
        self.last_decision_prompt = ""
        self.prompt_budget_info = {}
        # partner_context is configured by the runner and deliberately retained.

    def prepare_prompt(self, input_instruction, world_graph, **kwargs):
        instruction = (
            input_instruction + "\n\nIdentity and observation conventions:\n"
            "You are the robot, agent 0 (self). The human-shaped agent is your "
            "partner, agent 1. In ALL observations, including previous collaborations, "
            "held_by=0 means YOU hold the object; held_by=1 means YOUR PARTNER "
            "holds it; null means neither agent holds it. held_by_role states the "
            "same identity explicitly. robot_position is your position; human_position "
            "is your partner's position. recent_robot_action_results contains YOUR "
            "executed actions, not your partner's. current_observation is the latest "
            "state; recent_observed_changes contains earlier states labeled by "
            "sim_step. Do not attribute your own placements to your partner.\n"
            + "\nPartner information:\n"
            + self.partner_context
            + "\nCoordinate to complete the shared task efficiently. "
            "You can work on different unfinished objects."
        )
        return super().prepare_prompt(instruction, world_graph, **kwargs)

    def _add_responses_to_prompt(self, responses):
        # Keep the executed action and exact result, not every past world snapshot
        # or generated Thought. The next replan builds a fresh current scene.
        uid = self.agents[0].uid
        response = responses.get(uid, "")
        if not response:
            return ""
        self.action_results.append(
            {
                "action": self.last_high_level_actions.get(uid),
                "result": response,
            }
        )
        result = f"Result: {response}\n"
        self.trace += result
        return result

    def _reduce_optional_context(self):
        if self._turn_limit > 1:
            # Always preserve the latest action/result, particularly a failure.
            self._turn_limit = max(1, self._turn_limit // 2)
            return True
        if self._observation_limit:
            self._observation_limit = 0
            return True
        return False

    def replan(self, instruction, observations, world_graph):
        uid = self.agents[0].uid
        # Base contains the unchanged task, tool format, frozen partner context
        # and a fresh world description. Never append to the old curr_prompt.
        base, self.params = self.prepare_prompt(
            instruction, world_graph[uid], observations=observations
        )
        retries = 0
        while True:
            recent = (
                list(self.recent_observations)[-self._observation_limit :]
                if self._observation_limit
                else []
            )
            context = {
                "current_observation": self.public_observation,
                "recent_observed_changes": recent,
                "recent_robot_action_results": list(self.action_results)[
                    -self._turn_limit :
                ],
            }
            self.curr_prompt = (
                base
                + "\nLatest observable scene (not a partner plan):\n"
                + json.dumps(context, separators=(",", ":"))
                + "\nReturn Thought: followed by your reasoning, then one bare tool call such as Pick[toy_food_0] on its own line, then Assigned! on its own line. Do not prefix the tool call with Action:.\n"
            )
            self.last_decision_prompt = self.curr_prompt
            self.prompt_budget_info = {
                "prompt_chars": len(self.curr_prompt),
                "retained_action_results": min(
                    len(self.action_results), self._turn_limit
                ),
                "retained_observations": len(recent),
                "context_overflow_retries": retries,
            }
            if len(self.curr_prompt) > self.max_prompt_chars:
                if self._reduce_optional_context():
                    continue
                raise RuntimeError(
                    "Essential coordination prompt exceeds max_prompt_chars. "
                    "Task, frozen partner context, current observation and latest action result "
                    "were preserved. Reduce coordination.max_history_chars or use a larger "
                    "serving context and character guard."
                )
            try:
                return super().replan(instruction, observations, world_graph)
            except RuntimeError as error:
                if not is_context_overflow(error):
                    raise
                if not self._reduce_optional_context():
                    raise RuntimeError(
                        "Essential coordination prompt exceeds the server token budget even "
                        "after dropping older current-episode context. Frozen partner context "
                        "was NOT changed. Reduce coordination.max_history_chars or increase "
                        "the serving context; rerun all comparison conditions consistently. "
                        f"Original error: {error}"
                    ) from error
                retries += 1
