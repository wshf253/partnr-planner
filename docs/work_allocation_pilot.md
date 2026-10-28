# Work allocation: initial data pilot

This is a fixed-model failure-discovery pilot, not a training or continual-learning
experiment. Reuse PARTNR episode 99 as the development evaluation task: dispersed
objects go to two destinations. Keep the existing simulator, skills and success
predicates. Compare batch and nearest partners with no_history and frozen history.
The old three-partner experiment remains available unchanged through its config.

## Run

```bash
lab-run --new --gpus 1 --cpus 8 bash scripts/run_work_allocation_partnr.sh
```

Default: one automatically selected distinct support layout, episode 99 as test,
one seed, two partners and two conditions: 2 support + 4 evaluation rollouts.
Output budget is 2048. Support is recollected; previous episode-99 traces must not
be used as support for episode-99 evaluation. The split selector reserves the test
layout before selecting support and rejects duplicated physical layouts. Actual
support ID is recorded in split.json. Set `coordination.support_ids=['ID']` to pin
a different compatible layout after inspecting candidates; no ID is silently
substituted if the requested test episode is unavailable.

Optional diagnostic (8 rollouts including support):

```bash
lab-run --new --gpus 1 --cpus 8 bash scripts/run_work_allocation_partnr.sh \
  'coordination.conditions=[no_history,history,oracle_style]'
```

## Data and interpretation

Retain config.yaml, split.json, results.jsonl and all trial folders, including
successful trajectories. Existing partner_context.txt records the exact evidence
presented; public_trace.json and robot_decisions.jsonl preserve observations,
prompts, actions and results via events.jsonl.

New evaluator-private partner_choices.jsonl records every partner selection
boundary: step, actual profile/choice, current observation, last selected
destination, cooldown exclusions, Euclidean distances, and what batch/nearest
would select from that SAME state and last destination. Probing uses fresh policy
instances and does not alter either agent's behavior. No candidate choice or
private profile is included in public history or robot prompts.

results.jsonl adds partner_choice_points and partner_style_divergence_points.
These are local choice opportunities, not optimal-action labels, counterfactual
full rollouts, or proof that personalization helps. The last destination is the
last selected destination, not necessarily a successfully completed placement.
Distance is Euclidean, not navigable path cost. Repeated polls can contribute
multiple choice points; inspect the logged steps, not just aggregate counts.

```bash
python scripts/summarize_coordination.py PATH_TO_COORDINATION/results.jsonl
```

The summary includes support and test style_evidence_checks. Zero divergence means
this rollout did not expose a batch-versus-nearest choice opportunity; retain it
as such rather than treating it as evidence of failed personalization. Nonzero
divergence still needs manual inspection: the public support trace may not make
the pattern clear, and choosing the complementary object need not be optimal.
If support has no useful evidence, select another compatible support layout for
the next pilot. This implementation does not claim episode 99 is a validated
personalization task before simulator inspection.

Review duplicate commitments, partner-held execution failures (both agents),
repeated/unnecessary navigation and waiting versus reworking completed objects.
Preserve these as candidate failure annotations with observation/action timestamps.
The second proposed wait-versus-takeover task is not part of this pilot.
