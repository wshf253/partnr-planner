# Personalized takeover pilot

A controlled first scenario for preference adaptation in PARTNR. The human moves
one object to a target and pauses. The robot makes **one ACT/WAIT decision**:
ACT executes the remaining placement with the robot; WAIT lets the human resume
after a configurable number of simulator steps. Existing oracle Rearrange/Wait
skills perform all movement. There is no policy training.

## Scope and interpretation

- Full observation, two exact `is_on_top` goals on the same target. This is not
  a partial-observability or unrestricted multi-agent planning benchmark.
- The two simulated users differ in their evaluation preference and explicit
  post-episode feedback. Their pre-decision behavior is identical by design.
- `accept_help` prefers ACT; `keep_ownership` prefers WAIT. The private profile
  never enters the decision prompt except in the labeled oracle condition.
- The robot is idle until the decision point; after a choice, a protocol controls
  which agent owns the remaining object. This prevents simultaneous manipulation.
  It does not test online intervention cancellation or arbitrary replanning.
- Learned memory is a bounded history of observed choice/feedback pairs, not an
  LLM-written skill or a learned policy. It is an initial memory baseline.
- Every completed interaction provides explicit simulated feedback, including
  WAIT. This is deliberately informative; do not interpret improvement as evidence
  of implicit intention inference or real human personalization.
- No Ask action, preference drift, changing scenes or cross-task transfer yet.
- This version has unit and mocked runner tests. End-to-end Habitat/GPU rollout
  must be checked in the existing iiscluster installation.

## Files

| File | Purpose |
| --- | --- |
| `habitat_llm/personalization/core.py` | Protocol, strict episode validation, feedback memory and decision prompt |
| `habitat_llm/planner/takeover_skill_planner.py` | Executes assigned existing skills for either agent |
| `habitat_llm/evaluation/takeover_evaluation_runner.py` | Separates the LLM decision from scripted human execution |
| `habitat_llm/examples/personalization_demo.py` | Ordered condition/user/day loop, resets and output |
| `habitat_llm/conf/baselines/personalized_takeover.yaml` | Experiment settings and both agents |
| `habitat_llm/conf/planner/takeover_skill_planner.yaml` | Hydra class registration |
| `scripts/run_personalized_partnr.sh` | Reuse vLLM or start the existing Qwen launcher |

Two existing files change: `evaluation_runner.py` gains a no-op post-step hook;
`run_qwen35_partnr.sh` gains an opt-in experiment branch. Existing baseline
commands keep their behavior. The experiment uses a runner subclass directly;
do not run its config through `planner_demo.py`.

## Run on the configured cluster environment

From the repository root, inside an allocated GPU job using the same existing
`.venv`, simulator assets and Qwen installation as the baseline:

```bash
# Starts Qwen using the existing launcher and runs 2 users x 5 days x 3 conditions.
bash scripts/run_personalized_partnr.sh
```

The launcher inherits the current repository's model path and serving environment
from `run_qwen35_partnr.sh`. It shuts its own vLLM process down on exit.
If a service is already running, reuse it:

```bash
PARTNR_LLM_BASE_URL=http://127.0.0.1:18000/v1 \
  bash scripts/run_personalized_partnr.sh personalization.days=5
```

List supported episodes before running (requires the existing Habitat Python
environment and dataset, but does not construct a simulator or call an LLM):

```bash
.venv/bin/python -m habitat_llm.examples.personalization_demo \
  personalization.inspect_only=true
```

The default selects the first compatible episode in the dataset. It does **not**
assume index 0 is the phone/phone-stand task. Inspection prints exact episode IDs,
instructions and simulator handles. Pin the selected ID for reproducibility:

```bash
bash scripts/run_personalized_partnr.sh \
  personalization.episode_id=YOUR_EPISODE_ID \
  personalization.days=5 \
  personalization.pause_steps=120 \
  evaluation.save_video=true
```

Replace `YOUR_EPISODE_ID` with an ID printed by inspection. Optionally set
`personalization.first_object_handle` to one of the printed object handles to
choose which object the human moves first. Without that override, proposition
order determines the first object. Unsupported goals/constraints, alternative
receptacles and goals already satisfied at reset are rejected rather than silently
running a different task. If the selected episode is unsuitable, explicitly select
another compatible episode.

A short one-condition execution check:

```bash
bash scripts/run_personalized_partnr.sh \
  'personalization.conditions=[learned_memory]' personalization.days=2
```

`pause_steps` counts actual environment steps, not planner calls or LLM latency.
It does not use `Wait[duration]` because the current Wait skill ignores its argument.
This runner safely interrupts idle Wait at the pause deadline.

## Conditions and output

- `no_memory`: no past feedback in the prompt.
- `learned_memory`: last eight observed feedback records for this user/condition.
- `oracle_profile`: exact preferred response provided to the LLM.

Model, episode, physical initial seed and available actions are held fixed. Only
history differs. User IDs are not shown to the LLM. Each condition/user has separate
memory; fresh output directories prevent reuse from earlier experiments. The same
initial seed is reset each day; variation across days is intentionally absent in
this initial diagnostic (simulator determinism still needs runtime verification).

Under the Hydra run directory:

- `personalization/scenario.json`: selected episode and simulator handles.
- `personalization/config.yaml`: effective experiment config including private
  profiles, for evaluator reproducibility. This is not model input.
- `personalization/results.jsonl`: one record per trial, including decision,
  preference match, unwanted takeover choice, simulator steps, task success,
  completed flag, feedback, exact prompt and response.
- `personalization/<condition>/<user>/memory.json`: observed history only.
- Standard PARTNR planner logs/prompts/videos remain under `results/`.

`unwanted_takeover` measures an unwanted ACT choice, not successful physical
completion. Report it alongside `completed` and `task_state_success`.
A model-output error is an error, never an implicit WAIT. Navigation failures and
unfinished rollouts do not update memory. Exceptions are recorded and re-raised;
trial failures are not silently omitted. Aggregate choice metrics only over
trials with an actual decision, and report missing-decision and completion rates.

## Local checks without Habitat

```bash
python -m unittest discover -s tests -v
python -m compileall -q habitat_llm/personalization \
  habitat_llm/examples/personalization_demo.py \
  habitat_llm/evaluation/takeover_evaluation_runner.py \
  habitat_llm/planner/takeover_skill_planner.py
bash -n scripts/run_personalized_partnr.sh scripts/run_qwen35_partnr.sh
```

The mocked runner tests verify ownership, ACT and WAIT execution paths, pause clock,
single LLM invocation and failure handling. They do not validate Habitat imports,
Hydra composition against Habitat schemas, real skill success or GPU behavior.
