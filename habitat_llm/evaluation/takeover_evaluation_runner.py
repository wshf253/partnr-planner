"""Controlled takeover evaluation; hidden preferences never enter the decider."""

from habitat_llm.evaluation.decentralized_evaluation_runner import (
    DecentralizedEvaluationRunner,
)
from habitat_llm.personalization.core import (
    TakeoverProtocol,
    build_prompt,
    parse_decision,
)


class TakeoverEvaluationRunner(DecentralizedEvaluationRunner):
    def configure_trial(self, names, history, llm, pause_steps):
        self.names = names
        self.history = history
        self.llm = llm
        self.protocol = TakeoverProtocol(pause_steps)
        self.decision_prompt = ""
        self.decision_response = ""

    def on_environment_step(self, info):
        self.protocol.tick()

    def get_low_level_actions(self, instruction, observations, world_graph):
        protocol = self.protocol
        if protocol.phase == "pause" and not protocol.decision:
            self.decision_prompt = build_prompt(instruction, *self.names, self.history)
            self.decision_response = self.llm.generate(self.decision_prompt)
            protocol.choose(parse_decision(self.decision_response)["decision"])
        first, remaining, target = self.names
        actor = None
        obj = None
        if protocol.phase == "first":
            actor, obj = 1, first
        elif protocol.phase == "human_remaining":
            actor, obj = 1, remaining
        elif protocol.phase == "robot_remaining":
            actor, obj = 0, remaining
        for uid, planner in self.planner.items():
            planner.action = (
                ("Rearrange", f"{obj}, on, {target}, None, None", None)
                if uid == actor
                else ("Wait", "", None)
            )
        actions, info, _ = super().get_low_level_actions(
            instruction, observations, world_graph
        )
        if actor is not None and info["responses"].get(actor):
            protocol.skill_finished(info["responses"][actor])
        info["takeover"] = {"phase": protocol.phase, "decision": protocol.decision}
        # Standard logger expects a prompt for both agents if the key is present.
        info["prompts"] = {0: self.decision_prompt, 1: "Scripted human; no LLM prompt."}
        info["traces"] = {0: self.decision_response, 1: ""}
        finished = protocol.phase in ("done", "failed")
        return actions, info, finished
