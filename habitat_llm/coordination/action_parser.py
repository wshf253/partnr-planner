"""Accept an optional Action: label without changing the chosen skill or arguments."""

import re


def normalize_action_label(response):
    lines = response.strip().splitlines()
    if not lines:
        return response
    # Assigned! is normally consumed by the LLM stop sequence.
    if lines[-1].strip() == "Assigned!":
        lines.pop()
    while lines and not lines[-1].strip():
        lines.pop()
    if lines:
        match = re.fullmatch(
            r"Action:\s*([A-Za-z_][A-Za-z_0-9]*\[[^\[\]\r\n]*\])",
            lines[-1].strip(),
        )
        if match:
            lines[-1] = match.group(1)
    return "\n".join(lines)


def coordination_action_parser(agents, input_string, params=None):
    from habitat_llm.llm.instruct.utils import zero_shot_action_parser

    # The planner retains the raw response in its trace; normalize only parsing.
    return zero_shot_action_parser(agents, normalize_action_label(input_string), params)
