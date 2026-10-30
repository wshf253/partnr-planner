# Rule correctness diagnostic

Question: does useful coordination depend on a rule matching the actual partner,
or does either rule merely change the robot's task order?

```bash
lab-run --new --gpus 1 --cpus 8 bash scripts/run_rule_diagnostic_partnr.sh
```

Default: train episode 8958, batch and nearest partners, no_history/oracle_style/
opposite_style, seed 47668090, 5000 sim steps maximum. Six fresh isolated rollouts;
all initial observations and agent base poses must match before actions execute.
No memory generation, forced robot actions, or partner behavior changes.

For an actual batch partner, opposite_style supplies the exact nearest description;
for actual nearest, it supplies the exact batch description. The wording and prefix
are identical to the corresponding oracle description; the planner is not told
that the rule is incorrect. Evaluator results record actual profile and supplied_profile.
The actual partner policy always receives the actual profile.

candidate_comparison.json includes rule_comparisons: no_history -> oracle_style,
no_history -> opposite_style, and opposite_style -> oracle_style. Positive steps_saved
means the treatment completed sooner. Only successful, initially matched pairs
have timing comparisons; errors remain visible. Inspect actions as well as times.

If either supplied rule improves similarly, this does not support partner-specific
benefit. If correct outperforms opposite and no information, it is a signal worth
following up, not proof from a single seed. Wrong information hurting alone does
not establish that learning a partner improves performance. Retain all outcomes.

24318 and 24635 remain experience candidates; no new memory experiment is launched
until an evaluation situation is informative. To include the secondary evaluation
candidate in this same diagnostic, append `candidate_sweep.episode_ids='[8958,8990]'`
(12 rollouts). 8990 has prior output-token and scripted recovery failures, so it is
not enabled by default. This script starts a new run and does not resume old runs.
