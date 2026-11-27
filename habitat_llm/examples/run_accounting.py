#!/usr/bin/env python3
"""Failure accounting and run provenance for ``planner_demo``.

This module imports nothing from habitat, habitat-sim or torch, so it can be
unit tested without the simulator. ``planner_demo`` uses it only when the new
config keys ``evaluation.count_failures`` / ``evaluation.write_run_info`` are
set; with their defaults (False) the demo behaves as before.

It stays compatible with Python 3.9 (the PARTNR environment).
"""

import csv
import hashlib
import importlib
import json
import os
import platform
import re
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# LLM failure kinds raised by the backend adapter (``LLMCallFailed.kind``).
LLM_FAILURE_KINDS = ("http", "timeout", "length", "empty", "budget")
STATUSES = (
    ("ok",) + tuple(f"llm_{kind}" for kind in LLM_FAILURE_KINDS) + ("parse", "sim", "other")
)

# Episode CSV columns when failures are counted. Every row has the same
# columns, so failed rows (which have no simulator stats) stay aligned with
# successful ones. Further scalar stats go into ``other_stats`` as JSON.
EPISODE_COLUMNS = (
    "run_id",
    "episode_id",
    "instruction",
    "status",
    "failed_agent",
    "task_percent_complete",
    "task_state_success",
    "sim_step_count",
    "runtime",
    "other_stats",
)

RUN_INFO_FILE = "run_info.json"


###############################################################################
# Episode guard
###############################################################################


def check_episode_indices(num_proc: int, episode_indices: Optional[Sequence[int]]) -> None:
    """``planner_demo`` applies ``episode_indices`` only with ``num_proc == 1``.

    Before this check a run with ``num_proc > 1`` silently evaluated the whole
    dataset; now it is an error.
    """
    if episode_indices is not None and num_proc != 1:
        raise ValueError(
            f"episode_indices={list(episode_indices)} is only applied with num_proc=1, "
            f"but num_proc={num_proc}. Use num_proc=1, or drop episode_indices to run "
            "the whole dataset."
        )


###############################################################################
# Failure classification and aggregation
###############################################################################


def _exception_chain(exc: Optional[BaseException]) -> Iterable[BaseException]:
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        yield exc
        exc = exc.__cause__ or exc.__context__


def _is_simulator_exception(exc: BaseException) -> bool:
    module = type(exc).__module__ or ""
    root = module.split(".")[0]
    return root in ("habitat_sim", "magnum") or root == "habitat"


def classify_exception(exc: BaseException) -> Tuple[str, Optional[int]]:
    """Map an exception raised while running an episode to ``(status, agent_uid)``.

    LLM and parse failures are recognised by a ``kind`` attribute anywhere in
    the exception chain (the backend adapter raises ``LLMCallFailed(kind)``);
    the agent uid comes from an ``agent_uid`` attribute when the raiser knows
    it. Exceptions defined in habitat, habitat-sim or magnum count as ``sim``.
    Simulator errors raised as builtin exceptions from C++ bindings cannot be
    told apart and count as ``other``.
    """
    for item in _exception_chain(exc):
        kind = getattr(item, "kind", None)
        if kind in LLM_FAILURE_KINDS:
            return f"llm_{kind}", getattr(item, "agent_uid", None)
        if kind == "parse":
            return "parse", getattr(item, "agent_uid", None)
    for item in _exception_chain(exc):
        if _is_simulator_exception(item):
            return "sim", getattr(item, "agent_uid", None)
    return "other", None


def failed_episode_stats() -> Dict[str, float]:
    """Stats entered into the aggregates for an episode that raised.

    ``aggregate_measures`` averages each key over the episodes that have it,
    so a failed episode counts as 0 in ``task_percent_complete`` and
    ``task_state_success`` and is absent from the other means (steps, runtime).
    """
    return {"task_percent_complete": 0.0, "task_state_success": 0.0}


def status_summary(statuses: Iterable[str]) -> Dict[str, int]:
    """``n_episodes``, ``n_failed`` and ``n_<status>`` for every status."""
    counts = {status: 0 for status in STATUSES}
    total = 0
    for status in statuses:
        if status not in counts:
            raise ValueError(f"Unknown episode status {status!r}")
        counts[status] += 1
        total += 1
    summary = {"n_episodes": total, "n_failed": total - counts["ok"]}
    summary.update({f"n_{status}": count for status, count in counts.items()})
    return summary


def merge_status_runs(runs: Iterable[Mapping[str, Mapping[str, str]]]) -> Dict[str, Dict[str, str]]:
    """Merge per-process ``{run_id: {episode_id: status}}`` mappings."""
    merged: Dict[str, Dict[str, str]] = {}
    for run in runs:
        for run_id, episodes in run.items():
            merged.setdefault(str(run_id), {}).update(episodes)
    return merged


def all_statuses(status_runs: Mapping[str, Mapping[str, str]]) -> List[str]:
    return [status for episodes in status_runs.values() for status in episodes.values()]


def episode_row(
    run_id: int,
    episode_id: str,
    instruction: str,
    stats: Mapping[str, Any],
    status: str = "ok",
    failed_agent: Optional[int] = None,
) -> Dict[str, Any]:
    """One row of ``episode_result_log.csv`` in the fixed column layout."""
    row: Dict[str, Any] = {
        "run_id": run_id,
        "episode_id": episode_id,
        "instruction": instruction,
        "status": status,
        "failed_agent": "" if failed_agent is None else failed_agent,
    }
    for key in ("task_percent_complete", "task_state_success", "sim_step_count", "runtime"):
        row[key] = stats.get(key, "")
    others = {k: v for k, v in sorted(stats.items()) if k not in row}
    row["other_stats"] = json.dumps(others, sort_keys=True) if others else ""
    return row


def ensure_episode_csv(path: str) -> None:
    """Create ``path`` with the fixed header unless it already has content.

    Called once in the parent process, so worker processes only append.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, newline="") as f:
            header = next(csv.reader(f), [])
        if tuple(header) != EPISODE_COLUMNS:
            raise ValueError(
                f"{path} has columns {header}, not the failure-accounting layout "
                f"{list(EPISODE_COLUMNS)}. Use a new hydra.run.dir."
            )
        return
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(EPISODE_COLUMNS)


def append_episode_row(path: str, row: Mapping[str, Any]) -> None:
    with open(path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=EPISODE_COLUMNS).writerow(row)


###############################################################################
# Provenance (run_info.json)
###############################################################################

# Whole name parts only, so ``max_tokens`` or ``stop_tokens`` are kept.
_SECRET_NAME = re.compile(
    r"(^|_)(api_?key|token|secret|password|passwd|authorization|credentials?)($|_)"
)
REDACTED = "<redacted>"


def _is_secret_name(name: str) -> bool:
    name = name.lower().replace("-", "_")
    # ``api_key_env: ICRL_LLM_API_KEY`` names a variable; the name is not a secret.
    return bool(_SECRET_NAME.search(name)) and not name.endswith("_env")


def redact(obj: Any) -> Any:
    """Replace values under secret-looking keys, recursively."""
    if isinstance(obj, Mapping):
        return {
            k: (REDACTED if _is_secret_name(str(k)) and v not in (None, "") else redact(v))
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return [redact(v) for v in obj]
    return obj


def secret_values(environ: Mapping[str, str]) -> List[str]:
    """Values of secret-looking environment variables, longest first."""
    values = {v for k, v in environ.items() if _is_secret_name(k) and v and len(v) >= 8}
    return sorted(values, key=len, reverse=True)


def scrub_text(text: str, secrets: Sequence[str]) -> str:
    for value in secrets:
        text = text.replace(value, REDACTED)
    return text


def _git(repo: str, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", repo, *args], check=True, capture_output=True, timeout=60
    ).stdout


def git_state(repo: Optional[str]) -> Dict[str, Any]:
    """Commit, dirty flag and a hash over the commit plus uncommitted changes.

    ``code_hash`` covers HEAD, the diff of tracked files against HEAD, and the
    paths and contents of untracked files that git does not ignore. Two
    checkouts with the same ``code_hash`` run the same code.
    """
    if not repo:
        return {"path": None, "available": False}
    repo = os.path.abspath(os.path.expanduser(repo))
    try:
        head = _git(repo, "rev-parse", "HEAD").decode().strip()
        status = _git(repo, "status", "--porcelain").decode()
        diff = _git(repo, "diff", "HEAD", "--binary")
        untracked = [p for p in _git(repo, "ls-files", "--others", "--exclude-standard", "-z").decode().split("\0") if p]
    except (OSError, subprocess.SubprocessError) as exc:
        return {"path": repo, "available": False, "error": type(exc).__name__}
    digest = hashlib.sha256()
    digest.update(head.encode())
    digest.update(diff)
    for path in sorted(untracked):
        digest.update(b"\0" + path.encode() + b"\0")
        try:
            with open(os.path.join(repo, path), "rb") as f:
                for block in iter(lambda: f.read(1 << 20), b""):
                    digest.update(block)
        except OSError:
            digest.update(b"<unreadable>")
    return {
        "path": repo,
        "available": True,
        "commit": head,
        "dirty": bool(status.strip()),
        "code_hash": digest.hexdigest(),
    }


def package_versions(names: Sequence[str] = ("habitat_sim", "habitat", "habitat_baselines", "torch", "numpy", "hydra", "omegaconf")) -> Dict[str, Optional[str]]:
    """Versions of the given modules, via ``__version__`` or package metadata."""
    from importlib import metadata

    dists = {"habitat_sim": "habitat-sim", "habitat": "habitat-lab", "habitat_baselines": "habitat-baselines", "hydra": "hydra-core"}
    versions: Dict[str, Optional[str]] = {}
    for name in names:
        version = None
        try:
            version = metadata.version(dists.get(name, name))
        except metadata.PackageNotFoundError:
            try:
                version = getattr(importlib.import_module(name), "__version__", None)
            except Exception:
                version = None
        versions[name] = None if version is None else str(version)
    return versions


ENV_RECORDED = (
    "COOP_SERVING_VERSION",
    "COOP_CHAT_TEMPLATE_KWARGS",
    "COOP_PARTNER_CHAT_TEMPLATE_KWARGS",
    "ICRL_LLM_BASE_URL",
    "ICRL_LLM_MODEL",
    "ICRL_PARTNER_BASE_URL",
    "ICRL_PARTNER_MODEL",
    "PARTNR_LLM_BASE_URL",
    "LLM_MODEL_PATH",
)


def prompt_record(prompt: Optional[str]) -> Dict[str, Optional[str]]:
    if prompt is None:
        return {"sha256": None, "text": None}
    return {"sha256": hashlib.sha256(prompt.encode()).hexdigest(), "text": prompt}


def build_run_info(
    repo: str,
    coop_repo: Optional[str],
    agents: Mapping[str, Mapping[str, Any]],
    dataset_path: str,
    episode_ids: Sequence[str],
    environ: Mapping[str, str],
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Everything needed to say which code, model and prompts produced a run.

    ``agents`` maps an agent name to ``{"llm": <llm config dict>,
    "instruct": <instruct config dict>}``; the instruct config's ``prompt`` is
    recorded in full together with its hash. Secrets are redacted by key name
    here and by value when the file is written.
    """
    agent_info = {}
    for name, parts in agents.items():
        instruct = dict(parts.get("instruct") or {})
        prompt = instruct.pop("prompt", None)
        agent_info[name] = {
            "llm": redact(parts.get("llm")),
            "instruct": redact(instruct),
            "prompt_template": prompt_record(prompt),
        }
    info: Dict[str, Any] = {
        "partnr_planner": git_state(repo),
        "coop_repo": git_state(coop_repo),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions(),
        "environment": {k: environ.get(k) for k in ENV_RECORDED},
        "agents": agent_info,
        "dataset_path": dataset_path,
        "episode_ids": [str(e) for e in episode_ids],
    }
    if extra:
        info.update(redact(dict(extra)))
    return info


def code_hashes(info: Mapping[str, Any]) -> Dict[str, Optional[str]]:
    return {
        key: (info.get(key) or {}).get("code_hash")
        for key in ("partnr_planner", "coop_repo")
    }


class RunFolderError(RuntimeError):
    """The run folder does not match the requested run."""


def write_or_check_run_info(
    run_dir: str,
    info: Mapping[str, Any],
    resume: bool,
    has_previous_results: bool,
    environ: Mapping[str, str],
) -> str:
    """Write ``run_info.json`` once; on resume, refuse if the code changed.

    - New run, no ``run_info.json``: write it.
    - New run, ``run_info.json`` present: refuse (new runs use new folders).
    - Resume, ``run_info.json`` present: keep it if the code hashes match,
      refuse otherwise.
    - Resume, no ``run_info.json`` but earlier results: refuse, because the
      code of the earlier run is unknown. Without earlier results, write it.
    """
    path = os.path.join(run_dir, RUN_INFO_FILE)
    if os.path.exists(path):
        if not resume:
            raise RunFolderError(
                f"{path} exists. New runs use a new hydra.run.dir; pass resume=True to continue this one."
            )
        with open(path) as f:
            previous = json.load(f)
        old, new = code_hashes(previous), code_hashes(info)
        if old != new:
            raise RunFolderError(
                f"Refusing to resume {run_dir}: code changed since the run started "
                f"(run_info.json has {old}, the current checkout has {new})."
            )
        return path
    if resume and has_previous_results:
        raise RunFolderError(
            f"Refusing to resume {run_dir}: it has results but no {RUN_INFO_FILE}, "
            "so the code of the earlier run is unknown. Use a new hydra.run.dir."
        )
    os.makedirs(run_dir, exist_ok=True)
    text = scrub_text(json.dumps(info, indent=2, sort_keys=True, default=str), secret_values(environ))
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text + "\n")
    os.replace(tmp, path)
    return path
