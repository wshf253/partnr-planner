# CLAUDE.md

Notes for Claude Code (and other coding agents) working in this fork of
`facebookresearch/partnr-planner`.

## Read first

- [`docs/PARTNR_LABRUN_PLAN.md`](docs/PARTNR_LABRUN_PLAN.md): the plan for running PARTNR
  decentralized episodes (ego robot `agent_0`, LLM partner human `agent_1`) on the lab server
  with the same serving conventions, model, backend layer, failure accounting and provenance as
  the coop runs. Work proceeds in its order (W0 ... W6). W5 and W6 end with a report; stop there
  and wait for SJ's decision.
- The plan names `main` as the branch. Work on this fork currently happens on
  `feat/qwen38-table-partner` (SJ's decision, 2026-11-27). That branch also carries
  `table_partner/`, a repeated table-setting experiment with scripted partners and memory
  across days. It is outside the plan's scope (plan sections 1 and 14); do not build plan
  work on top of it and do not change it as part of plan steps.

## Invariants (plan section 2, short form)

1. Upstream baselines and `run_qwen35_partnr.sh` keep their behaviour. `local_chat.py` and
   `local_chat.yaml` stay unchanged. New behaviour only through new files and new config keys
   whose defaults reproduce the current code. Exception: `episode_indices` with `num_proc > 1`
   is an error.
2. API keys only from environment variables or `~/.coop_api_env` (mode 600). Never in configs,
   resolved Hydra configs, logs, traces or errors.
3. Tests make no network calls; use a local fake HTTP server.
4. Infrastructure failures are recorded per episode and per agent with their cause. Never drop
   episodes after seeing outcomes.
5. New runs use new output folders. A resume with different code is refused.
6. Record model ids, serving version, decoding settings, backend specs (without keys), and the
   hash and full text of every prompt template in the run folder.

## Lab server

- The user runs experiments on the lab cluster (`ssh iiscluster`). The repo is expected at
  `~/work/partnr-planner` (confirm with the user). The PARTNR environment is the repo's `.venv`;
  the vLLM environment is `~/work/llm-serving/.venv-qwen35`. Do not suggest conda: the lab
  forbids conda installs and conda environments on the servers.
- GPUs are only visible inside Slurm jobs, started with `lab-run` (`lab-run --new ...` for a
  one-off job, or inside an interactive job from `lab-gpu start`). The login session has a
  limit of 8 CPU threads and 24 GiB RAM; tests, builds and runs do not go there.
- habitat-sim needs a GPU (EGL), so every job in this repo, including tests and heuristic or
  mock runs, uses at least `--gpus 1`. There are no `--gpus 0` jobs here. (To confirm once on
  the server that a `--gpus 0` job cannot create the habitat-sim context, then note it here.)
- LLM jobs go through `scripts/run_qwen38_partnr.sh` (plan W1, not written yet). When other GPU
  jobs run at the same time (the coop runs), use
  `lab-run --new --spare --gpus 1 --cpus 24 --mem 110G`. One GPU's guaranteed share is
  14 CPU threads, so 24 CPUs needs `--spare` (preemptible).
- `lab-run` passes the login session's environment variables into the job (lab guide,
  "Interactive Job"). Options go before the command and are never abbreviated (`--gpus`, not
  `--gpu`). Whether `lab-run` accepts a `--` separator before the command is not confirmed;
  check `lab-run --help` before relying on it.
- `outputs/` is git-ignored and so not in the nightly code backup. Results worth keeping are
  copied to `~/ckpt/<run name>/`.

## Companion repository

The backend layer lives in Collab-Overcooked (`src/coop/llm.py`, branch
`claude/lucid-babbage-96q0zi`, default checkout `$HOME/work/collab-overcooked`). Changes to it
are made there, on its research branch, with its tests and its old-config equality check. Do
not vendor a modified copy into this repo.

## Tests

- Simulator-free tests run anywhere:
  `python -m pytest habitat_llm/tests/test_run_accounting.py` (its aggregation test is skipped
  outside the PARTNR environment) and `python -m unittest table_partner.test_scenario`.

## Run accounting keys (plan W4)

`planner_demo` reads two optional keys, both off by default (old behaviour):

- `evaluation.count_failures`: episodes that raise stay in the results with a `status`
  (`habitat_llm/examples/run_accounting.py`), count as 0 in `task_percent_complete` and
  `task_state_success`, and the run and end logs get `n_episodes`, `n_failed` and `n_<status>`.
  The episode CSV then has a fixed column layout.
- `evaluation.write_run_info`: writes `run_info.json` into `hydra.run.dir` once; a new run in a
  used folder, or a resume with different code, is refused.

On a baseline config that does not define them, set them with `++evaluation.count_failures=True`
(Hydra struct configs reject plain overrides of unknown keys).
- The rest of `habitat_llm/tests` needs the `hssd-partnr-ci` data and a GPU, so it runs on the
  lab server under `lab-run --new --gpus 1`.

## Replies to the user

End every update with the commands to pull the branch and the `lab-run` commands to run, each
in its own `bash` block.
