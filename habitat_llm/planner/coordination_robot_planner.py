"""Existing ReAct action space with frozen partner context and fresh observations."""

import json

from habitat_llm.planner.zero_shot_react_planner import ZeroShotReactPlanner


class CoordinationRobotPlanner(ZeroShotReactPlanner):
    partner_context = "No previous experience."
    public_observation = None
    recent_observations = ()
    last_decision_prompt = ""

    def prepare_prompt(self, input_instruction, world_graph, **kwargs):
        instruction = (
            input_instruction
            + "\n\nPartner information:\n"
            + self.partner_context
            + "\nCoordinate to complete the shared task efficiently. "
            "You can work on different unfinished objects."
        )
        return super().prepare_prompt(instruction, world_graph, **kwargs)

    def replan(self, instruction, observations, world_graph):
        context = {
            "current_observation": self.public_observation,
            "recent_observed_changes": list(self.recent_observations),
        }
        self.curr_prompt += (
            "\nLatest observable scene (not a partner plan):\n"
            + json.dumps(context)
            + "\n"
        )
        if len(self.curr_prompt) > self.max_prompt_chars:
            raise RuntimeError(
                "Prompt budget exceeded; reduce history or increase serving context and max_prompt_chars"
            )
        self.last_decision_prompt = self.curr_prompt
        return super().replan(instruction, observations, world_graph)
