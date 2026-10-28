"""Pure-Python protocol, memory and episode validation for the takeover pilot."""

import json
from dataclasses import dataclass
from pathlib import Path

PROFILES = {
    "accept_help": {
        "ACT": "Thank you. When I pause like this, please finish the remaining work.",
        "WAIT": "Next time I pause like this, please finish the remaining work for me.",
    },
    "keep_ownership": {
        "ACT": "I wanted to finish that myself. When I pause like this, please wait.",
        "WAIT": "Thank you for waiting. When I pause like this, I want to finish it myself.",
    },
}


def parse_decision(text):
    """Reject malformed model output rather than silently turning it into WAIT."""
    text = text.strip()
    if text.startswith("```json\n") and text.endswith("```"):
        text = text[8:-3].strip()
    value = json.loads(text)
    if not isinstance(value, dict) or value.get("decision") not in ("ACT", "WAIT"):
        raise ValueError("Expected a JSON object with decision ACT or WAIT")
    if not isinstance(value.get("reason", ""), str):
        raise ValueError("reason must be a string")
    return value


class FeedbackMemory:
    """Append observed feedback only. No private profile enters this store."""

    def __init__(self, path):
        self.path = Path(path)
        self.entries = json.loads(self.path.read_text()) if self.path.exists() else []

    def context(self):
        if not self.entries:
            return "No previous interaction feedback is available."
        return "\n".join(
            f"Day {e['day']}: chose {e['decision']}; user said: {e['feedback']}"
            for e in self.entries[-8:]
        )

    def record(self, day, decision, feedback):
        self.entries.append(dict(day=day, decision=decision, feedback=feedback))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.entries, indent=2) + "\n")
        temporary.replace(self.path)


def build_prompt(instruction, first_object, remaining_object, target, history):
    # Deliberately omit user IDs, private profile, evaluator state and future actions.
    return (
        "You are a robot collaborating with a person. Both objects must reach the "
        "target. The person has moved the first object and is now pausing. "
        "They have not explicitly delegated the remaining object. Decide whether "
        "to take over or wait, using any feedback from previous interactions.\n"
        f"Task: {instruction}\n"
        f"Observed: {first_object} has been moved to {target}. "
        f"{remaining_object} still needs to be moved there. The person is pausing.\n"
        f"Previous interaction feedback:\n{history}\n"
        "ACT means move the remaining object now. WAIT means let the person resume.\n"
        'Return only JSON: {"decision": "ACT" or "WAIT", "reason": "short reason"}'
    )


def placement_pair(episode):
    """Accept exactly two unambiguous on-top goals, sharing one target.

    Accept two single-object propositions or one proposition requiring both objects.
    Reject richer tasks rather than silently changing the benchmark semantics.
    """
    if episode.get("evaluation_proposition_dependencies"):
        raise ValueError("Proposition dependencies are not supported in this pilot")
    for constraint in episode.get("evaluation_constraints", []):
        kind, args = constraint["type"], constraint.get("args", {})
        if kind == "TemporalConstraint" and not args.get("dag_edges", []):
            continue
        if kind == "TerminalSatisfactionConstraint":
            continue
        raise ValueError(f"Unsupported constraint: {kind}")
    pairs = []
    for prop in episode["evaluation_propositions"]:
        args = prop["args"]
        objects = args.get("object_handles", [])
        targets = args.get("receptacle_handles", [])
        if (
            prop["function_name"] != "is_on_top"
            or len(targets) != 1
            or args.get("number", 1) != len(objects)
            or not objects
        ):
            raise ValueError("Need two exact is_on_top goals to one target")
        pairs.extend((obj, targets[0]) for obj in objects)
    if len(pairs) != 2 or pairs[0][0] == pairs[1][0] or pairs[0][1] != pairs[1][1]:
        raise ValueError("Need two distinct objects and the same target")
    return pairs


@dataclass
class TakeoverProtocol:
    """Clock advances only after env.step; one explicit choice per interaction."""

    pause_steps: int
    phase: str = "first"
    sim_steps: int = 0
    pause_started: int = 0
    decision: str = ""
    error: str = ""

    def __post_init__(self):
        if self.pause_steps <= 0:
            raise ValueError("pause_steps must be positive")

    def tick(self):
        self.sim_steps += 1
        if self.phase == "pause" and self.decision == "WAIT":
            if self.sim_steps - self.pause_started >= self.pause_steps:
                self.phase = "human_remaining"

    def choose(self, decision):
        if self.phase != "pause" or self.decision:
            raise ValueError("Only one choice is allowed, at the pause")
        if decision not in ("ACT", "WAIT"):
            raise ValueError("Invalid decision")
        self.decision = decision
        if decision == "ACT":
            self.phase = "robot_remaining"

    def skill_finished(self, response):
        if self.phase not in ("first", "human_remaining", "robot_remaining"):
            raise ValueError("Unexpected manipulation completion")
        if response != "Successful execution!":
            self.phase, self.error = "failed", response
        elif self.phase == "first":
            self.phase, self.pause_started = "pause", self.sim_steps
        else:
            self.phase = "done"

    def result(self, profile):
        if profile not in PROFILES:
            raise ValueError(f"Unknown profile {profile}")
        complete = self.phase == "done"
        valid = bool(self.decision)
        preferred = "ACT" if profile == "accept_help" else "WAIT"
        # Explicit simulator feedback after the episode, not a claim of human data.
        feedback = PROFILES[profile][self.decision] if complete else None
        return {
            "phase": self.phase,
            "decision": self.decision or None,
            "preference_match": int(self.decision == preferred) if valid else None,
            "unwanted_takeover": (
                int(self.decision == "ACT" and preferred == "WAIT") if valid else None
            ),
            "sim_steps": self.sim_steps,
            "error": self.error or None,
            "feedback": feedback,
        }
