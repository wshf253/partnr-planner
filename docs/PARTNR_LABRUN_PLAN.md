# Plan: PARTNR on the lab server (`lab-run`), aligned with the coop setting

Project: self-evolving collaborative agent (ICML 2027).
Repo: `wshf253/partnr-planner` (fork of `facebookresearch/partnr-planner`), branch `main`.
Companion repo: `Collab-Overcooked`, branch `claude/lucid-babbage-96q0zi` (`CLAUDE.md`,
`scripts/run_qwen38_days.sh`, `scripts/run_api.sh`, `src/coop/llm.py`, `configs/coop/api/`).
For Claude Code. This repo has no `CLAUDE.md`; W0 adds one.

Scope: run PARTNR decentralized episodes (ego robot + LLM partner human) on the lab server with
the same serving script conventions, model, backend layer, failure accounting and provenance as
the coop runs. Persona partners, multi-episode memory and day structure are out of scope
(section 9).

## 0. Current code

The fork differs from upstream in four files:

| File | Content |
|---|---|
| `habitat_llm/llm/local_chat.py` | `LocalChat(BaseLLM)`: POST `/chat/completions`, transcript sent as one user message, `chat_template_kwargs={"enable_thinking": False}` hard-coded, no retries, raises on HTTP error, empty reply or `finish_reason == "length"`, raises `ValueError` if grammar arguments are passed |
| `habitat_llm/conf/llm/local_chat.yaml` | `base_url` from `PARTNR_LLM_BASE_URL`, model `qwen35-27b`, `max_tokens` 1024, temperature 0 |
| `scripts/run_qwen35_partnr.sh` | Serves `/lab/ckpt/jsj412/Qwen3.5-27B` with vLLM (`--max-num-seqs 1`, `--enforce-eager`, `--gpu-memory-utilization 0.75`, `--max-model-len 16384`); runs **`baselines/centralized_zero_shot_react_summary`** on `val_mini`, `episode_indices=[0]`, `num_proc=1`; output `outputs/qwen35/<timestamp>_<job id>` |
| `constraints-iiscluster.txt` | torch 2.7.1, numpy 1.26.4, ... for the lab venv (`.venv`) |

Problems for our setting:

1. **Centralized planner.** One LLM plans for both agents. Not an ego/partner setting.
2. **Model and serving** differ from coop: Qwen3.5-27B, `qwen35-27b`, fixed serving flags, endpoint
   variable `PARTNR_LLM_BASE_URL` instead of `ICRL_LLM_BASE_URL`, no partner endpoint.
3. **Decentralized base config is incompatible with `local_chat`.**
   `decentralized_zero_shot_react_summary.yaml` sets `constrained_generation: True` and
   `inference_mode: rlm` for both agents. `LLMPlanner.replan` then passes `generation_args`, and
   `LocalChat.generate` raises `ValueError`, so every episode fails.
4. **Failed episodes are dropped silently.** `planner_demo.py` (episode loop, around L385) catches
   the exception, prints a traceback and skips the episode; it never enters `stats_episodes`, so
   the aggregate covers only episodes that finished. Any LLM error (HTTP, length, empty reply)
   removes the episode from the mean.
5. **`episode_indices` is ignored when `num_proc > 1`** (`planner_demo.py` L178: the subset is
   taken only in the `num_proc == 1` branch).
6. **No API backend.** `openai_chat.py` uses `AzureOpenAI` only; it cannot reach Astra or Opus.
7. **No resume or provenance.** A new timestamped folder per run; nothing records code versions,
   model, serving version or prompt templates.

Facts about the planner that the design relies on:

- The decentralized runner calls the two planners one after the other inside one process, so a
  process has at most one LLM call in flight.
- With `agent_asymmetry: True`, only the robot's world graph is updated by the human's actions
  (`planner.py`, `update_by_other_agent_action`). The ego sees the partner only through effects:
  objects moved or placed, "held by the other agent", object states. The partner's high-level
  actions and position are not in the ego's prompt.
- The prompt is one growing ReAct transcript (`Thought: ... <action> Assigned!` +
  observations), stop word `Assigned!`, up to `replanning_threshold` calls per agent.
- `local_chat` sends the transcript as a chat user message, not as a raw completion. Results are
  therefore not comparable with the PARTNR paper's numbers; they are comparable across our
  conditions.

## 1. Design

| Item | Choice |
|---|---|
| Roles | `agent_0` (Spot robot) = ego; `agent_1` (humanoid) = partner |
| Planner | Decentralized zero-shot ReAct (`llm_zero_shot_react_planner`, `zero_shot_prompt`), `constrained_generation: False` for both agents |
| Local model | Qwen3.8-27B, served name `qwen3.8-27b`, thinking off, temperature 0 |
| Backend layer | `coop.llm` (stdlib only), imported from the Collab-Overcooked checkout, through a thin `BaseLLM` adapter. Same retry, concurrency limit, cost ledger, budget stop and `backends.yaml` registry as the coop runs |
| Endpoints | Ego: `ICRL_LLM_*`; partner: `ICRL_PARTNER_*`, defaulting to the ego's server (as in `run_qwen38_days.sh`) |
| Communication | None (PARTNR default) |

## 2. Invariants

1. Upstream baselines and the old script keep their behaviour. `local_chat`, `local_chat.yaml` and
   `run_qwen35_partnr.sh` stay unchanged. New behaviour comes only through new files and new config
   keys whose defaults reproduce the current code. One exception: W4 makes `episode_indices` with
   `num_proc > 1` an error instead of being silently ignored.
2. API keys come only from environment variables (or `~/.coop_api_env`, mode 600, as in coop's
   `run_api.sh`). They never appear in configs, resolved Hydra configs, logs, traces or errors.
   Add a test that runs the adapter against a fake server with a key set and searches every
   written file for it.
3. Tests make no network calls; use a local fake HTTP server.
4. Infrastructure failures are recorded per episode and per agent with their cause, and reported
   next to every result. Never drop episodes after seeing outcomes.
5. New runs use new output folders. A resume with different code is refused.
6. Record model ids, serving version, decoding settings, backend specs (without keys), and a hash
   plus the full text of every prompt template in the run folder.

## 3. W0: `CLAUDE.md` for this repo

Mirror the relevant parts of the coop `CLAUDE.md`:

- Read this plan first.
- The user runs experiments on the lab server (repo at `~/work/partnr-planner`; confirm the
  path) with `lab-run`. End every update with the commands to pull the branch and the `lab-run`
  commands to run, each in its own `bash` block.
- habitat-sim needs a GPU (EGL), so every job, including tests and heuristic or mock runs, uses
  at least `--gpus 1`; there are no `--gpus 0` jobs here. LLM jobs go through
  `scripts/run_qwen38_partnr.sh`. When other GPU jobs run at the same time (the coop runs), use
  `lab-run --new --spare --gpus 1 --cpus 24 --mem 110G`.
- The backend code lives in Collab-Overcooked (`src/coop/llm.py`). Changes to it are made there,
  on its research branch, with its tests and its old-config equality check.

## 4. W1: serving script `scripts/run_qwen38_partnr.sh`

Port of coop's `run_qwen38_days.sh`. Usage:

```
bash scripts/run_qwen38_partnr.sh <baseline config name> [hydra overrides ...]
```

- Variables and defaults, as in coop unless noted: `LLM_SERVE_ENV`
  (`$HOME/work/llm-serving/.venv-qwen35`), `LLM_MODEL_PATH` (`/lab/ckpt/jsj412/Qwen3.8-27B`),
  `ICRL_LLM_MODEL` (`qwen3.8-27b`), `MAX_NUM_SEQS` (default 8; must be at least `num_proc`), `GPUS`,
  `PARALLEL` (`dp` | `tp`), `GPU_MEM` (**default 0.75**: habitat-sim shares the GPU), `MAX_MODEL_LEN`
  (default 16384), `REASONING_PARSER` (`qwen3`), `LANGUAGE_MODEL_ONLY` (1), `ENFORCE_EAGER` (default 0;
  the fork used 1, so keep the switch).
- `COOP_REPO` (default `$HOME/work/collab-overcooked`); prepend `$COOP_REPO/src` to `PYTHONPATH`.
- With `ICRL_LLM_BASE_URL` already set, start no server and use that endpoint. With
  `SERVE_QWEN=0`, start no server either (both agents on API backends); the job still needs a
  GPU for habitat-sim.
- Optional split layout `SPLIT_GPU=1` (with `lab-run --gpus 2`): vLLM runs with
  `CUDA_VISIBLE_DEVICES=0`; the planner gets `habitat.simulator.habitat_sim_v0.gpu_device_id=1`.
  In this layout `GPU_MEM` may go back to 0.85.
- Partner endpoint: `ICRL_PARTNER_BASE_URL`, `ICRL_PARTNER_MODEL`, `ICRL_PARTNER_API_KEY`,
  `COOP_PARTNER_CHAT_TEMPLATE_KWARGS` default to the ego's values, exactly as in coop.
- API keys: source `${COOP_API_ENV:-~/.coop_api_env}` with the same mode check as coop's
  `run_api.sh`, unless the keys are already exported.
- Thinking check: `coop.check --pick` cannot be reused (its probe expects the Overcooked JSON
  `wait 1`). Add `habitat_llm/llm/check_backend.py`: build the ego backend with `coop.llm`, send
  one short ReAct prompt for each candidate in coop's `CANDIDATES`, pass when
  `LLMResult.thinking` is false and the reply contains an action line. Export the first passing
  value as `COOP_CHAT_TEMPLATE_KWARGS`. Write the probe results to the run folder. Do the same
  for the partner endpoint when it differs.
- Require an explicit `hydra.run.dir=...` (exit with a message if missing). Export
  `COOP_SERVING_VERSION` (vLLM version and model path) for W4.
- Then run `python -m habitat_llm.examples.planner_demo --config-name <config> "$@"`.

## 5. W2: backend adapter

### 5.1 `habitat_llm/llm/coop_backend.py`

`CoopBackendLLM(BaseLLM)`:

- `__init__(conf)`: `coop.llm.build_backend(conf.backend, name=conf.role)`. The import error
  message names `COOP_REPO`.
- `generate(prompt, stop=None, max_length=None, generation_args=None)`:
  - `generation_args` → raise a clear `ValueError` (grammar-constrained generation is not
    supported; set `constrained_generation=False`);
  - messages: system message from the config, then the transcript as one user message (as
    `local_chat`);
  - `max_tokens` = `max_length` or the config value;
  - per-call seed from (episode id, agent uid, call index);
  - pass `stop` to the backend once 5.3 exists; in any case truncate the reply at the first
    occurrence of the stop word (the planner appends `Assigned!` itself);
  - on `error`, empty text or `finish_reason == "length"`, raise `LLMCallFailed(kind)` with kind
    in `http | timeout | length | empty | budget`.
- Per-call log, one JSON line per call to `<run dir>/llm_calls/agent_<uid>.jsonl`: episode id,
  call index, prompt / completion / cached / reasoning tokens, latency, finish reason, cost,
  retries, error kind. Ego and partner usage stay separate.
- Check the PARTNR venv's Python version (likely 3.9 for habitat-sim 0.3.3) and that `coop.llm`
  imports under it. If it does not, report which construct fails; do not vendor a modified copy.

### 5.2 Configs

- `habitat_llm/conf/llm/coop_ego.yaml`: `_target_` the adapter; `role: ego`;
  `backend: {type: openai_compat, base_url_env: ICRL_LLM_BASE_URL, model_env: ICRL_LLM_MODEL,
  api_key_env: ICRL_LLM_API_KEY, chat_template_kwargs_env: COOP_CHAT_TEMPLATE_KWARGS,
  json_mode: false, params: {temperature: 0, top_p: 1}}`; `max_tokens: 1024`; system message and
  tags as in `local_chat.yaml`.
- `habitat_llm/conf/llm/coop_partner.yaml`: the same with `role: partner` and the `ICRL_PARTNER_*`
  / `COOP_PARTNER_CHAT_TEMPLATE_KWARGS` variables.
- API egos by override, for example
  `evaluation.agents.agent_0.planner.plan_config.llm.backend='{ref: astra, json_mode: false}'`.
  The registry's `astra` has `json_mode: true`; PARTNR replies are not JSON.

### 5.3 `stop` in `coop.llm` (Collab-Overcooked, separate commit)

`Backend.chat(..., stop=None)`. `openai_compat` and `openai` send `stop`; `anthropic` sends
`stop_sequences`. With `stop=None` payloads are byte-identical to the current ones (coop's mock
equality check and backend tests must pass unchanged). Whether Astra accepts `stop` is checked in
W6; if not, the adapter's truncation is the only mechanism for that backend, and its wasted
output tokens are reported.

## 6. W3: baseline config `habitat_llm/conf/baselines/decentralized_coop.yaml`

- Defaults as `decentralized_zero_shot_react_summary.yaml`, plus
  `/llm@evaluation.agents.agent_0.planner.plan_config.llm: coop_ego` and
  `/llm@evaluation.agents.agent_1.planner.plan_config.llm: coop_partner`.
- Both agents: `constrained_generation: False`, `replanning_threshold: 50`,
  `objects_response: True`; drop `inference_mode`.
- `world_model.partial_obs: True`, `agent_asymmetry: True` (open point, section 8).
- `evaluation.save_video: False`, `evaluation.log_detailed_traces: True`,
  `evaluation.count_failures: True` (W4).
- `habitat.dataset.data_path` default `data/datasets/partnr_episodes/v0_0/val_mini.json.gz`.
- Test: the config composes, both agents resolve to the adapter with the right role, and the
  grammar path is off.

## 7. W4: failure accounting, output folder, provenance (`planner_demo.py`)

- New key `evaluation.count_failures` (default `False` = current behaviour). With `True`:
  - an exception in `run_instruction` writes a row to `episode_result_log.csv` with `status`
    (`ok | llm_http | llm_timeout | llm_length | llm_empty | llm_budget | parse | sim | other`),
    the agent uid when known, `task_percent_complete = 0`, `task_state_success = 0`;
  - aggregates include these rows, and `end_result_log.csv` gets `n_episodes`, `n_failed` and
    the count per status;
  - successful rows get `status = ok`.
  Keep the accounting in a small function so it is unit-testable without the simulator.
- `episode_indices` with `num_proc > 1`: raise `ValueError` (today it is silently ignored).
- `run_info.json` in `hydra.run.dir`, written once: git hash and dirty flag of partnr-planner and
  of `COOP_REPO`, Python and package versions (habitat-sim, torch), `COOP_SERVING_VERSION`,
  `chat_template_kwargs` for each endpoint, both backend specs without keys, hash and full text
  of each agent's instruct prompt template, dataset path, the episode ids of the run.
- With `resume=True`, refuse to continue if the code hashes in `run_info.json` differ (as coop's
  `code_version.json`).

## 8. W5: smoke run and resource measurement

1. One episode, `num_proc=1`: check that both agents call their endpoints, the traces contain
   both agents' transcripts, `llm_calls/*.jsonl` and `run_info.json` are written, and a forced
   LLM failure (`MAX_MODEL_LEN` set very low in a second run) shows up as a failed row.
2. A few `val_mini` episodes with `num_proc=1`, then with `num_proc` 2 and 4. Record:
   - per-process GPU memory (`nvidia-smi`) and RSS;
   - vLLM memory at the chosen `GPU_MEM`;
   - maximum prompt tokens per agent and per episode, against `MAX_MODEL_LEN`;
   - LLM calls per agent per episode, wall time per episode, failures by status.
3. From these, propose `num_proc`, `MAX_NUM_SEQS`, `GPU_MEM`, `MAX_MODEL_LEN` and the GPU layout
   (shared or split) under `--cpus 24 --mem 110G`.

**Report, then stop.** SJ chooses the settings before any full split is run.

## 9. W6: API ego smoke (after SJ has seen the W5 report)

- Capture 3 prompts from the W5 traces (early, mid, late transcript) into
  `habitat_llm/tests/fixtures/`.
- For `astra` and `opus` (`json_mode: false`): send each fixture, report whether `stop` is
  accepted, the reply format (thought + one action line), input / cached / output / reasoning
  tokens, cost per call, and the projected cost per episode from the W5 call counts.
- Report whether the growing user message is read from the prompt cache. coop's Anthropic
  backend caches only the system block, and PARTNR's system message is short; if caching is
  absent, say what a `cache_control` breakpoint on the user message would save. Do not add it
  without SJ's decision.

**Report, then stop.**

## 10. Tests

- Adapter against a fake OpenAI-compatible server: message construction, truncation at the stop
  word, per-call log fields, `LLMCallFailed` kinds, grammar arguments rejected, and the key-leak
  search (invariant 2).
- Config composition test (W3).
- Failure accounting function with synthetic episode results, including a run where every
  episode fails.
- `episode_indices` with `num_proc > 1` raises.
- The existing `habitat_llm/tests` still pass on the lab server (they need the `hssd-partnr-ci`
  data and a GPU).

## 11. Order of work

| Step | Work | Output |
|---|---|---|
| W0 | `CLAUDE.md` | One commit |
| W1 | `scripts/run_qwen38_partnr.sh`, `check_backend.py` | One commit |
| W2 | Adapter, `coop_ego.yaml`, `coop_partner.yaml`, tests; `stop` in `coop.llm` as a separate Collab-Overcooked commit | Commits in both repos |
| W3 | `decentralized_coop.yaml`, composition test | One commit |
| W4 | Failure accounting, `episode_indices` guard, `run_info.json`, resume check, tests | One commit |
| W5 | Smoke run and measurements | **Report, then stop** |
| W6 | API ego smoke | **Report, then stop** |

After each step: tests pass, upstream baselines are unchanged, and the reply ends with the pull
and `lab-run` commands, each in its own `bash` block. Commands after W4, for reference (replace
`<branch>`; the `num_proc` and `MAX_NUM_SEQS` values are placeholders until W5):

```bash
cd ~/work/partnr-planner
```
```bash
git fetch origin && git switch <branch> && git pull --ff-only origin <branch>
```
```bash
lab-run --new --gpus 1 -- bash scripts/run_qwen38_partnr.sh baselines/decentralized_coop '++episode_indices=[0]' hydra.run.dir=outputs/partnr_smoke_qwen38
```
```bash
lab-run --new --spare --gpus 1 --cpus 24 --mem 110G -- env MAX_NUM_SEQS=8 GPU_MEM=0.75 bash scripts/run_qwen38_partnr.sh baselines/decentralized_coop num_proc=8 resume=True hydra.run.dir=outputs/partnr_valmini_qwen38
```

## 12. To verify, not assume

- The repo path on the lab server, and that the existing `.venv` has habitat-sim working under
  `lab-run` (EGL inside the job).
- That a `--gpus 0` job indeed cannot create the habitat-sim context (expected; confirm once and
  record it in `CLAUDE.md`).
- The PARTNR venv's Python version and the `coop.llm` import (5.1).
- Whether `lab-run` passes exported environment variables into the job.
- Per-process GPU memory and RSS, the maximum prompt length, and whether `ENFORCE_EAGER=0` starts
  cleanly beside habitat-sim (W5).
- `stop` support and prompt caching on the API backends (W6).

## 13. Open points for SJ

- `agent_asymmetry`: `True` (PARTNR default; only the robot's world graph gets the human's action
  results) or `False` (the partner also gets the ego's action results, so it can react to the
  ego beyond its own perception).
- Failed episodes in the main metric: counted as 0 (the W4 default with `count_failures`) or
  excluded with `n_failed` reported beside the result.
- GPU layout: one shared GPU or `SPLIT_GPU=1` with two GPUs (decided from W5).

## 14. After this plan (not to implement now)

1. **Partner persona.** `agent_1` uses the same `zero_shot_prompt` as the robot plus the PARTNR
   "task giver" role sentence. A persona needs its own instruct template or planner subclass for
   `agent_1`.
2. **Memory across episodes and day structure.** `eval_runner.reset()` resets the planners after
   every episode. Days (several interaction episodes with the same partner, day-end feedback,
   consolidation, held-out evaluation) need a runner on top of `run_planner`, as `coop.days`.
3. **What the ego observes of the partner.** Today only world-graph effects. Inferring the
   partner from its behaviour may require exposing its high-level actions within a visibility
   range, the PARTNR counterpart of coop's `view_radius`.
4. **Episode sets.** PARTNR episodes differ in scene and instruction. Day-level learning with one
   partner needs an episode set per day (for example a fixed scene set with varied instructions)
   and a held-out set.
