# Candidate screening for partner learning and planning

Run the ten previous train candidates with batch/nearest crossed with
no_history/oracle_style: 40 trials, one existing seed, maximum 5000 sim steps each.
No support memory collection or forced robot action is used.

```bash
lab-run --new --gpus 1 --cpus 8 bash scripts/run_candidate_sweep_partnr.sh
```

Use `candidate_sweep.episode_ids='[24318,8990]'` to select explicit episode IDs.
Each trial gets a fresh environment and independent episode copy. The existing
pre-action check verifies initial public observations and agent base transforms
against the first trial for the episode/seed. A mismatched trial does not execute;
its error is retained, and screening continues with other trials. Setup-invalid
candidates are also retained as errors. Missing/incompatible IDs fail at selection.

Outputs in the usual coordination directory:
- results.jsonl: raw metrics/errors, per-trial log directory and robot action list.
- candidate_comparison.json: automatically refreshed after every trial. Positive
  steps_saved_by_oracle means fewer completion steps with the true rule supplied.
  Only successful, initially matched pairs are compared. Errors and timeouts stay visible.
- Per-trial public_trace.json, partner_choices.jsonl, robot_decisions.jsonl,
  events.jsonl, and initial_state_check.json for inspection.

public_traces_differ is only a heuristic: differences may reflect timing, robot
responses, or duration, not necessarily identifiable partner traits. Use observable
partner behavior to shortlist experience tasks. Use execution outcomes and actions
to shortlist evaluation tasks. Oracle is neither an optimal policy nor a ceiling;
zero improvement does not automatically disqualify a candidate. No optimum or
causal benefit of learned memory is claimed by this screen.

8990 previously exposed a scripted partner recovery problem (retrying an object
held by itself); preserve and inspect those failures separately from ToM failures.
21204/21424 previously had too few unfinished goals and may be rejected again.
Do not mix prior unisolated-run timing metrics with these comparisons.
