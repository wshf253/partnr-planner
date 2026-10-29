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
