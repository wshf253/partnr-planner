# Continual partner memory pilot

Goal: test whether accumulated partner evidence changes coordination and completion time.
No weight training, forced robot allocations, hidden profile labels in learned memory,
or evaluation feedback in updates. This is an RQ1/RQ3 pilot, not an RQ2 result.

Run from the repository:

```bash
lab-run --new --gpus 1 --cpus 8 bash scripts/run_continual_partner_partnr.sh
```

Defaults: train episode 3 provides experience; episode 24318 is an evaluation probe.
The existing split validator requires different physical layouts. Batch and nearest
partners have independent memories. Three rounds evaluate zero, one, and two previous
experiences. Each round resets the probe for no_history, memory, fixed_memory, and
oracle_style. fixed_memory retains the first inferred memory. Day 1 has no memory;
Day 2 memory and fixed_memory are identical controls. There are 24 probe rollouts
and 4 experience rollouts. No repeated branch-validation runs are needed.

One shared experience stream per partner is generated using the evolving memory.
Probe outcomes never update memory. Thus comparisons isolate access to the same
acquired memory, not independently deployed learning algorithms. The experience
layout repeats: gains may reflect layout familiarity, so this small pilot cannot
establish generalization or robust continual learning. The oracle describes the
true partner rule but is not an optimal planner or guaranteed performance ceiling.

Inspect results.jsonl and each experience directory's memory_before.txt,
memory_update_prompt.txt, memory_update_raw.txt, memory_after.txt, and public_trace.json.
Compare success first, then sim_steps on matched successful pairs; inspect action
traces to connect changed plans to outcomes. Collision reduction alone is not RQ3.
Errors remain in results; support/update errors abort instead of silently freezing
memory. No measured improvement is guaranteed. Keep negative outcomes.

RQ2 comes later: distinguish transient constraints from persistent strategy changes,
with observations that allow the distinction across time. Do not infer it from this
stable-partner experiment.

Memory budgets: default maximum 6000 characters, generation target 4000 characters,
and output budget 2048 tokens. Oversized memories are re-summarized up to twice;
text is never silently sliced. Raw generations, summary prompts/outputs, and
memory_budget.json preserve the audit trail. Exhausted retries and empty outputs
raise distinct errors. Override continual.memory_max_chars,
continual.summary_target_chars, continual.memory_max_tokens, and
continual.summary_retries via Hydra. These character limits do not replace the
model context-token limit. Existing runs are not resumed by this script.

Fairness update: each continual rollout creates a fresh simulator/task/planner and
an independent episode copy with seeds set before construction. Before any action,
the initial public observation and both agent base transforms (rounded to 6 decimals)
are checked against the first rollout for that episode/seed, across days, partners,
and conditions. Mismatch or rollout error aborts the sequence and saves diagnostics;
no mismatched comparison is accepted. This checks observed state and agent poses,
not bitwise equality of all physics internals. Simulator construction adds overhead;
compare sim_steps, not initialization wall time.

Public observations now include cumulative partner travel and empty-handed moving
steps (>0.005 displacement per step). These are measured from observed positions,
not private partner action labels or intentions. The memory prompt explicitly warns
that empty hands do not imply idleness and stationarity does not identify a trait.
Re-run all conditions together: old and new observation formats are not comparable.
