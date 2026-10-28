# Partner-dependent coordination failure diagnostics

This experiment asks whether the **same fixed robot planner needs different
coordination strategies for different partners**, and whether past observations
or a partner-style description help. It does not train models, learn a memory
updater, or label ACT/WAIT as correct. It supersedes the simple takeover task for
failure discovery, while keeping that pilot runnable.

## Protocol

1. Find existing PARTNR tasks with at least four exact on-top placement goals,
   including at least two destination groups containing two objects each.
   The executed instruction is regenerated from those goals as a shared task,
   allowing either agent to do any placement. Original corpus role assignments
   are not used; physical layout and supported evaluation predicates are retained.
2. Select distinct physical layouts for support interactions and held-out tests.
   Automatic selection does not fall back to reusing the test layout.
3. For each partner/seed, collect support interactions using the no-history robot.
4. Freeze the observed support records. Run every held-out task with:
   - `no_history`: current episode only;
   - `history`: observations from those fixed support interactions;
   - `oracle_style`: the partner's behavioral rule, but no private current target
     or future action sequence.
5. Reset both planners and the scene for each test condition. Test interactions
   never update support history, model weights or skills.

With defaults: **3 partners x (1 support + 1 test x 3 conditions) = 12 rollouts**.
A default one-support/one-test pair is a small diagnostic, not a generalization
benchmark. Use multiple held-out episodes and seeds after inspecting the first
trajectories. A given layout may not distinguish the partner types at all; that
is a result to inspect, not a guaranteed failure.

## Partner behaviors

| Profile | Rule at skill boundaries |
| --- | --- |
| `batch` | Start with a nearby object, then finish available objects for that destination before switching destinations. |
| `nearest` | Choose the nearest available unfinished object by Euclidean distance each time, ignoring destination grouping. |
| `yielding` | Start from nearest choice but avoid objects when the robot is closer, within `yield_radius`, and visibly approaching; wait if all are avoided. |

All skip completed goals and objects carried by the robot. A failed human skill
causes a short retry cooldown for that object, allowing other work to continue.
The partner polls while waiting. It never reads the robot's private action target
and never interrupts an in-progress manipulation. Thus `yielding` is boundary
level yielding, not instantaneous mutual avoidance. Grouping is by destination,
not object category. Distances are straight-line, not obstacle-aware travel costs.

The robot uses the existing **ZeroShotReactPlanner** and existing motor skills.
It can select Rearrange, Navigate, Pick, Place, Wait, etc.; it is not restricted to
ACT/WAIT. It replans when skills return, as in the base planner. No runner assigns
its next object, prevents duplicate selections, or forces a supposedly optimal
role. Done stops the robot's contribution; the human may continue until team
success or a step limit. Shared success is measured by Habitat.

## Observation and information boundaries

This version intentionally uses **full observation**. Current object locations,
holding states, goal destinations and both agents' positions are available to the
robot. No hidden human target or private profile is in the no-history/history
prompt. Public histories include the task, initial scene, periodic observations,
observed holding/location changes and final success. They exclude the human's
chosen targets, private policy state, robot hidden reasoning and evaluator
metrics. Oracle-style context is explicitly labeled separately.

The history is an unlearned, deterministic serialization of observed experience,
not an LLM-generated summary. If necessary, intermediate observations are
uniformly downsampled to fit `max_history_chars`, keeping endpoints. Exact context
is saved in each trial. No explicit preference feedback or correctness labels are
generated. Support traces can contain failures and uninformative behavior.

Physical-layout fingerprints exclude goal wording and episode IDs; they use scene
and object transforms/state metadata. Initial observation hashes also check paired
test resets. A false `paired_initial_state_matches` requires inspecting/resetting
before interpreting differences as effects of partner context.

## Run

Use the existing allocated GPU environment and installed `.venv`/Habitat assets.
First inspect the available tasks **without starting vLLM or a simulator**:

```bash
.venv/bin/python -m habitat_llm.examples.coordination_demo \
  coordination.inspect_only=true
```

Inspection still requires the installed Habitat Python dependencies because they
register Hydra schemas. It prints candidate IDs, instructions, goals and layout
fingerprints. Default `val_mini.json.gz` may not contain enough compatible tasks.
If so, point `habitat.dataset.data_path` at a larger dataset that is already
installed. Do not reduce the task to one destination or reuse a support layout in
test just to make the command run.

```bash
# Selects the first two compatible distinct layouts, starts the existing Qwen
# service, then runs all three partner types and all three conditions.
bash scripts/run_coordination_partnr.sh

# Pin IDs returned by inspection; replace SUPPORT_ID and TEST_ID below.
bash scripts/run_coordination_partnr.sh \
  'coordination.support_ids=[SUPPORT_ID]' \
  'coordination.test_ids=[TEST_ID]' \
  evaluation.save_video=true

# Reuse an already-running local endpoint instead of starting another server.
PARTNR_LLM_BASE_URL=http://127.0.0.1:18000/v1 \
  bash scripts/run_coordination_partnr.sh
```

For a larger diagnostic, select more compatible layouts or explicit lists:

```bash
bash scripts/run_coordination_partnr.sh \
  coordination.support_count=2 coordination.test_count=3 \
  'coordination.seeds=[47668090,47668091]'
```

This example runs 66 rollouts. There is no assumption that an episode ID is its
index. Use `coordination_demo`, not `planner_demo`, for this config. The launcher
reuses the existing Qwen model/service paths and cleans up its own service.
The default robot planning threshold is 60 decisions, with 5000 simulator steps
per trial. ReAct keeps its existing skill-boundary cadence; no mid-skill interruption
mechanism is introduced. At every decision the prompt is rebuilt from the task,
frozen partner context, current world state and at most six recent action/results
and three observations; old Thoughts and scene snapshots do not accumulate.
An explicit vLLM input-context rejection triggers bounded retries dropping older
current-episode records, while preserving the latest action/result and the entire
frozen partner context. Other server/model errors are not retried. If essential
context still cannot fit, the trial fails explicitly. `prompt_budget` in decision
logs records retained history and retry count. This bounded prompt policy applies
to all conditions; rerun all conditions when comparing to pre-fix results.

## Outputs to inspect or share

Under the Hydra run directory, `coordination/` contains:

- `results.jsonl`: support and test rows, failures, success, steps and diagnostic
  counts. Filter `phase == "test"` for comparison.
- `config.yaml`, `split.json`: settings and verified support/test layout split.
- A folder per trial, containing:
  - `partner_context.txt`: exact frozen context supplied for this condition;
  - `robot_decisions.jsonl`: each robot prompt, current observation, thought and
    chosen action, with simulator timestamps;
  - `public_trace.json`: only observable trajectory data, eligible for support;
  - `events.jsonl`: evaluator-only action/result events, including human actions.
- Standard PARTNR logs/videos remain in the usual `results/` subdirectory of the
  Hydra run directory. The custom trace replaces the base mixed-agent pickle
  action history; that base recorder assumes all agents are LLM planners.

Start by sharing `results.jsonl`, `split.json`, and the selected trial's
`robot_decisions.jsonl` + `events.jsonl`. Videos help distinguish planning errors
from locomotion failures.

```bash
python scripts/summarize_coordination.py /path/to/coordination/results.jsonl
```

The summary preserves execution errors and compares completion steps **only when
both paired trials succeed and their initial-state hashes match**. Timeouts are
not counted as fast completions.

## Metrics and interpretation

- `task_state_success`, `task_percent_complete`: authoritative Habitat outcomes.
- `sim_steps`: simulated duration; `runtime` also includes inference and logging.
- `same_object_target_steps`: both active high-level skills refer to the same goal
  object. A review signal, not proof of physical collision or wasted motion.
- `robot_attempts_held_by_human` / `robot_attempts_already_at_goal`: suspicious new
  selections, not guaranteed semantic errors (Navigate can have other uses).
- `robot_idle_with_available_work_steps`: robot Wait/Done while an unheld unfinished
  goal exists. It may be strategic waiting; inspect the scene.
- `robot_distance`, `human_distance`: accumulated geometric displacement.
- `robot_skill_failures`, `human_skill_failures`: separate execution/tool errors.
- `human_yield_count`: candidate exclusions at human selection times, not episodes.
- `robot_finished_before_team`: robot declared Done or exhausted its planning
  budget while team success had not yet been observed.

Graph furniture membership is a proxy for pending work. It is not the exact
on-top predicate; physics/task success remains authoritative. Episodes with too
little unfinished work at reset are rejected. Errors are logged and raised rather
than silently skipped. Prompt overflow is also an execution error; character
limits are guards, not exact tokenizer budgets.

Useful candidate findings: oracle-style improves over no-history on a matched
layout; history still fails where oracle-style succeeds; or history helps on one
held-out layout and misleads on another. If all conditions fail physically, inspect
skill failures first. If all perform alike, inspect whether partner behavior ever
diverged before claiming the planner is robust. This implementation does not test
partner switching, preference drift, partial observation or continual adaptation
yet; it establishes whether a partner-dependent failure exists first.

## Validation

Pure-Python tests cover distinct partner choices, observed-motion yielding,
held-object exclusion, split leakage rejection, frozen public histories and
mixed-runner termination/metrics. Planner Hydra composition is checked separately.
Actual Habitat schema composition, scene assets and GPU rollouts still require
the cluster environment.

```bash
python -m unittest discover -s tests -v
bash -n scripts/run_coordination_partnr.sh scripts/run_qwen35_partnr.sh
```
