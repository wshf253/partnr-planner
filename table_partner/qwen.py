"""Robot decision at the intervention point, using an OpenAI-compatible server."""
import json
import os
import re
import urllib.error
import urllib.request

MODEL = "qwen3.8-27B"
CHECKPOINT = "/lab/ckpt/jsj412/Qwen3.8-27B"
PROMPT = (
    "You are setting the table repeatedly with the same person. They started carrying plates. "
    "Choose join to take another plate if they are likely to switch to cups. Choose separate "
    "to carry a clean cup if they are likely to keep carrying plates. Learn from observed "
    "responses at similar remaining workloads. You have no other information about the person. "
    'Return exactly one JSON object {"decision":"join"} or {"decision":"separate"}.'
)


def choose_qwen(observation):
    root = (os.getenv("QWEN_BASE_URL") or os.getenv("PARTNR_LLM_BASE_URL")
            or "http://127.0.0.1:8010/v1").rstrip("/")
    model = os.getenv("QWEN_MODEL", MODEL)
    messages = [{"role": "system", "content": PROMPT},
                {"role": "user", "content": json.dumps(observation)}]
    headers = {"Content-Type": "application/json"}
    key = os.getenv("QWEN_API_KEY")
    if key:
        headers["Authorization"] = "Bearer " + key
    for attempt in range(2):
        payload = dict(model=model, messages=messages, temperature=0, max_tokens=1024,
                       chat_template_kwargs={"enable_thinking": False})
        req = urllib.request.Request(root + "/chat/completions", json.dumps(payload).encode(), headers)
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                content = json.load(response)["choices"][0]["message"]["content"]
        except (OSError, KeyError, IndexError, ValueError) as exc:
            raise RuntimeError(f"Qwen request failed at {root} (model {model}): {exc}") from exc
        try:
            content = re.sub(r"^\s*<think>.*?</think>\s*", "", content or "", flags=re.S)
            answer = json.loads(content)
            if isinstance(answer, dict) and answer.get("decision") in ("join", "separate"):
                return answer["decision"]
        except (TypeError, ValueError):
            pass
        if attempt == 0:
            messages.append({"role": "user", "content": "Reply with only the requested JSON object."})
    raise RuntimeError("Qwen returned an invalid decision twice")


def choose_reference(obs):
    """Nearest-workload behavioral baseline used to check the protocol without a GPU."""
    if not obs["history"]:
        return "separate"
    load = obs["plates_remaining"] + obs["cups_remaining"]
    distance = min(abs(e["remaining"] - load) for e in obs["history"])
    nearest = [e for e in obs["history"] if abs(e["remaining"] - load) == distance]
    return "join" if sum(e["response"] == "switch" for e in nearest) > len(nearest)/2 else "separate"
