#!/usr/bin/env python3
"""Failure accounting, the episode_indices guard and run provenance (plan W4).

Simulator-free: only the aggregation test needs habitat_llm's evaluation
functions and is skipped where they cannot be imported.
"""

import csv
import json
import os
import subprocess

import pytest

from habitat_llm.examples.run_accounting import (
    EPISODE_COLUMNS,
    REDACTED,
    RunFolderError,
    all_statuses,
    append_episode_row,
    build_run_info,
    check_episode_indices,
    classify_exception,
    ensure_episode_csv,
    episode_row,
    failed_episode_stats,
    merge_status_runs,
    redact,
    status_summary,
    write_or_check_run_info,
)

SECRET = "sk-test-0123456789abcdef"


class FakeLLMCallFailed(RuntimeError):
    def __init__(self, kind, agent_uid=None):
        super().__init__(f"LLM call failed: {kind}")
        self.kind = kind
        self.agent_uid = agent_uid


# ---------------------------------------------------------------------------
# episode_indices guard


def test_episode_indices_with_several_processes_raises():
    with pytest.raises(ValueError, match="num_proc=1"):
        check_episode_indices(4, [0, 1])


@pytest.mark.parametrize("num_proc,indices", [(1, [0]), (1, None), (4, None)])
def test_episode_indices_allowed(num_proc, indices):
    check_episode_indices(num_proc, indices)


# ---------------------------------------------------------------------------
# classification


@pytest.mark.parametrize("kind", ["http", "timeout", "length", "empty", "budget"])
def test_llm_failures_keep_kind_and_agent(kind):
    assert classify_exception(FakeLLMCallFailed(kind, agent_uid=1)) == (f"llm_{kind}", 1)


def test_llm_failure_found_through_wrapping_exception():
    try:
        try:
            raise FakeLLMCallFailed("length", agent_uid=0)
        except FakeLLMCallFailed as inner:
            raise RuntimeError("planner failed") from inner
    except RuntimeError as outer:
        assert classify_exception(outer) == ("llm_length", 0)


def test_parse_sim_and_other():
    parse = FakeLLMCallFailed("parse", agent_uid=1)
    sim_error = type("SimError", (RuntimeError,), {"__module__": "habitat_sim.simulator"})
    assert classify_exception(parse) == ("parse", 1)
    assert classify_exception(sim_error("crash")) == ("sim", None)
    assert classify_exception(KeyError("x")) == ("other", None)
    # An unknown kind is not an LLM failure.
    assert classify_exception(FakeLLMCallFailed("weird")) == ("other", None)


# ---------------------------------------------------------------------------
# summaries and aggregation


def test_status_summary_counts_every_status():
    summary = status_summary(["ok", "ok", "llm_http", "sim"])
    assert summary["n_episodes"] == 4
    assert summary["n_failed"] == 2
    assert summary["n_ok"] == 2
    assert summary["n_llm_http"] == 1
    assert summary["n_sim"] == 1
    assert summary["n_llm_budget"] == 0


def test_status_summary_when_every_episode_fails():
    summary = status_summary(["llm_timeout"] * 3)
    assert (summary["n_episodes"], summary["n_failed"], summary["n_ok"]) == (3, 3, 0)


def test_status_summary_rejects_unknown_status():
    with pytest.raises(ValueError):
        status_summary(["ok", "crashed"])


def test_merge_status_runs_from_processes():
    merged = merge_status_runs([{"0": {"1": "ok"}}, {"0": {"2": "llm_http"}}, {1: {"1": "ok"}}])
    assert merged == {"0": {"1": "ok", "2": "llm_http"}, "1": {"1": "ok"}}
    assert sorted(all_statuses(merged)) == ["llm_http", "ok", "ok"]


def test_failed_episodes_count_as_zero_in_aggregates():
    evaluation_functions = pytest.importorskip(
        "habitat_llm.agent.env.evaluation.evaluation_functions",
        reason="needs the PARTNR environment",
    )
    stats = {
        "1": {"task_percent_complete": 1.0, "task_state_success": 1.0, "sim_step_count": 100.0},
        "2": failed_episode_stats(),
    }
    metrics = evaluation_functions.aggregate_measures(stats)
    assert metrics["task_percent_complete"] == pytest.approx(0.5)
    assert metrics["task_state_success"] == pytest.approx(0.5)
    assert metrics["sim_step_count"] == pytest.approx(100.0)

    all_failed = evaluation_functions.aggregate_measures(
        {str(i): failed_episode_stats() for i in range(3)}
    )
    assert all_failed["task_state_success"] == 0.0


# ---------------------------------------------------------------------------
# episode CSV


def test_episode_rows_stay_aligned(tmp_path):
    path = str(tmp_path / "results" / "episode_result_log.csv")
    ensure_episode_csv(path)
    ok = {
        "task_percent_complete": 1.0,
        "task_state_success": 1.0,
        "sim_step_count": 120.0,
        "runtime": 3.5,
        "replanning_count_0": 4.0,
    }
    append_episode_row(path, episode_row(0, "7", "set the table", ok))
    append_episode_row(
        path,
        episode_row(0, "8", "clear, the table", failed_episode_stats(), "llm_length", 1),
    )
    ensure_episode_csv(path)  # existing header is kept

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == list(EPISODE_COLUMNS)
    assert rows[0]["status"] == "ok" and rows[0]["failed_agent"] == ""
    assert json.loads(rows[0]["other_stats"]) == {"replanning_count_0": 4.0}
    assert rows[1]["status"] == "llm_length"
    assert rows[1]["failed_agent"] == "1"
    assert rows[1]["instruction"] == "clear, the table"
    assert rows[1]["task_state_success"] == "0.0"
    assert rows[1]["sim_step_count"] == ""


def test_episode_csv_with_old_layout_is_refused(tmp_path):
    path = tmp_path / "episode_result_log.csv"
    path.write_text("episode_id,task_state_success\n1,1.0\n")
    with pytest.raises(ValueError, match="new hydra.run.dir"):
        ensure_episode_csv(str(path))


# ---------------------------------------------------------------------------
# provenance


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    path = tmp_path / "repo"
    path.mkdir()
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "test@example.com")
    _git(path, "config", "user.name", "test")
    (path / "planner.py").write_text("print('v1')\n")
    _git(path, "add", "planner.py")
    _git(path, "commit", "-q", "-m", "v1")
    return path


def _info(repo, environ):
    agents = {
        "agent_0": {
            "llm": {
                "backend": {"api_key_env": "ICRL_LLM_API_KEY", "api_key": SECRET},
                "max_tokens": 1024,
                "generation_params": {"temperature": 0.0, "model": "qwen3.8-27b"},
            },
            "instruct": {"prompt": "You are an agent. {tool_descriptions}", "stopword": "Assigned!"},
        }
    }
    return build_run_info(
        repo=str(repo),
        coop_repo=None,
        agents=agents,
        dataset_path="data/datasets/partnr_episodes/v0_0/val_mini.json.gz",
        episode_ids=["0", "3"],
        environ=environ,
    )


def test_redact_keeps_env_names_and_token_limits():
    redacted = redact(
        {"api_key": SECRET, "api_key_env": "ICRL_LLM_API_KEY", "max_tokens": 9, "HF_TOKEN": "x" * 10}
    )
    assert redacted == {
        "api_key": REDACTED,
        "api_key_env": "ICRL_LLM_API_KEY",
        "max_tokens": 9,
        "HF_TOKEN": REDACTED,
    }


def test_run_info_records_provenance_without_keys(tmp_path, repo):
    environ = {
        "ICRL_LLM_API_KEY": SECRET,
        "COOP_SERVING_VERSION": "vllm 0.0.0 /lab/ckpt/model",
        # A key that also shows up in a recorded value is scrubbed by value.
        "ICRL_LLM_BASE_URL": f"http://127.0.0.1:1/v1?key={SECRET}",
    }
    run_dir = tmp_path / "run"
    path = write_or_check_run_info(str(run_dir), _info(repo, environ), False, False, environ)

    for root, _dirs, files in os.walk(run_dir):
        for name in files:
            with open(os.path.join(root, name)) as f:
                assert SECRET not in f.read()

    with open(path) as f:
        info = json.load(f)
    agent = info["agents"]["agent_0"]
    assert agent["llm"]["max_tokens"] == 1024
    assert agent["llm"]["backend"]["api_key_env"] == "ICRL_LLM_API_KEY"
    assert agent["prompt_template"]["text"] == "You are an agent. {tool_descriptions}"
    assert len(agent["prompt_template"]["sha256"]) == 64
    assert agent["instruct"] == {"stopword": "Assigned!"}
    assert info["environment"]["COOP_SERVING_VERSION"] == "vllm 0.0.0 /lab/ckpt/model"
    assert info["partnr_planner"]["dirty"] is False
    assert info["coop_repo"]["available"] is False
    assert info["episode_ids"] == ["0", "3"]


def test_new_run_in_used_folder_is_refused(tmp_path, repo):
    run_dir = str(tmp_path / "run")
    write_or_check_run_info(run_dir, _info(repo, {}), False, False, {})
    with pytest.raises(RunFolderError, match="new hydra.run.dir"):
        write_or_check_run_info(run_dir, _info(repo, {}), False, False, {})


def test_resume_with_same_code_continues(tmp_path, repo):
    run_dir = str(tmp_path / "run")
    path = write_or_check_run_info(run_dir, _info(repo, {}), False, False, {})
    before = open(path).read()
    write_or_check_run_info(run_dir, _info(repo, {}), True, True, {})
    assert open(path).read() == before


@pytest.mark.parametrize("change", ["edit", "untracked", "commit"])
def test_resume_with_different_code_is_refused(tmp_path, repo, change):
    run_dir = str(tmp_path / "run")
    write_or_check_run_info(run_dir, _info(repo, {}), False, False, {})
    if change == "edit":
        (repo / "planner.py").write_text("print('v2')\n")
    elif change == "untracked":
        (repo / "new_module.py").write_text("x = 1\n")
    else:
        (repo / "planner.py").write_text("print('v2')\n")
        _git(repo, "commit", "-q", "-am", "v2")
    with pytest.raises(RunFolderError, match="code changed"):
        write_or_check_run_info(run_dir, _info(repo, {}), True, True, {})


def test_resume_of_folder_without_provenance_is_refused(tmp_path, repo):
    run_dir = str(tmp_path / "run")
    with pytest.raises(RunFolderError, match="no run_info.json"):
        write_or_check_run_info(run_dir, _info(repo, {}), True, True, {})
    # A resume flag on an empty folder simply starts the run.
    write_or_check_run_info(run_dir, _info(repo, {}), True, False, {})
