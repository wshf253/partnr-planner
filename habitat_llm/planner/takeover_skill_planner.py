"""Execute protocol-selected skills without introducing new motor policies."""

from habitat_llm.planner.planner import Planner


class TakeoverSkillPlanner(Planner):
    def __init__(self, plan_config, env_interface):
        super().__init__(plan_config, env_interface)
        self.action = ("Wait", "", None)
        self.active = None

    def reset(self):
        for agent in self.agents:
            agent.reset()
        self.active = None
        self.last_high_level_actions = {}

    def get_next_action(self, instruction, observations, world_graph):
        uid = self.agents[0].uid
        if self.active != self.action:
            # The protocol only interrupts idle Wait, never a manipulation.
            if self.active is not None and self.active[0] != "Wait":
                raise RuntimeError("Cannot interrupt an active manipulation")
            self.agents[0].reset()
            self.active = self.action
        self.last_high_level_actions = {uid: self.action}
        actions, responses = self.process_high_level_actions(
            self.last_high_level_actions, observations
        )
        if responses.get(uid):
            self.active = None
        return (
            actions,
            {
                "high_level_actions": dict(self.last_high_level_actions),
                "responses": responses,
            },
            False,
        )
