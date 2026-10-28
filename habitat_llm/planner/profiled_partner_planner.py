"""Scripted partner selects goals from observable geometry at skill boundaries."""

from habitat_llm.coordination.core import PartnerPolicy, partner_choice_audit
from habitat_llm.coordination.observations import public_observation
from habitat_llm.planner.takeover_skill_planner import TakeoverSkillPlanner


class ProfiledPartnerPlanner(TakeoverSkillPlanner):
    def configure(self, profile, goals, poll_steps, yield_radius):
        self.profile, self.goals = profile, goals
        self.poll_steps, self.yield_radius = poll_steps, yield_radius
        self.reset()

    def reset(self):
        super().reset()
        self.policy = PartnerPolicy(
            getattr(self, "profile", "batch"), getattr(self, "yield_radius", 2.0)
        )
        self.blocked_until = {}
        self.wait_started = 0
        self.skill_failures = 0
        self.last_choice_audit = None

    def get_next_action(self, instruction, observations, world_graph):
        self.last_choice_audit = None
        if self.active is not None and self.active[0] == "Wait":
            if self.sim_step - self.wait_started >= self.poll_steps:
                self.agents[0].reset()
                self.active = None
        if self.active is None:
            blocked = [
                obj
                for obj, until in self.blocked_until.items()
                if until > self.sim_step
            ]
            audit = partner_choice_audit(
                self.goals, self.scene, self.policy.group, blocked
            )
            goal = self.policy.select(self.goals, self.scene, blocked)
            self.last_choice_audit = {
                "sim_step": self.sim_step,
                "profile": self.profile,
                "selected_goal": goal.__dict__.copy() if goal else None,
                "observation": public_observation(self.scene, self.sim_step),
                **audit,
            }
            self.action = (
                ("Rearrange", f"{goal.object}, on, {goal.target}, None, None", None)
                if goal is not None
                else ("Wait", "", None)
            )
            self.wait_started = self.sim_step
        actions, info, _ = super().get_next_action(
            instruction, observations, world_graph
        )
        response = info["responses"].get(1, "")
        if (
            response
            and self.action[0] == "Rearrange"
            and response != "Successful execution!"
        ):
            self.skill_failures += 1
            obj = self.action[1].split(",", 1)[0].strip()
            self.blocked_until[obj] = self.sim_step + self.poll_steps * 4
        return actions, info, False
