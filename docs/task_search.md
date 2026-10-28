# Automatic task screening

Run compatibility inspection without starting an LLM server (Habitat Python is
still needed for the demo imports):

```bash
.venv/bin/python -m habitat_llm.examples.coordination_demo \
  --config-name baselines/partner_task_search coordination.inspect_only=true
```

Run up to 10 distinct compatible layouts with batch and nearest partners, no
history/support, one seed, capped at 2500 simulator steps per rollout:

```bash
lab-run --new --gpus 1 --cpus 8 bash scripts/run_task_search_partnr.sh
```

For a small first batch use `search.limit=5`. A larger dataset can be supplied via
`habitat.dataset.data_path=PATH_TO_DATASET.json.gz`. The default val_mini may have
far fewer than 10 compatible layouts. The program reports the actual count;
it never manufactures candidates. Selection is reproducible hashed sampling
(selection_seed), not a geometric suitability score. Compatibility is restricted
to exact independent placements, >=4 objects and >=2 destinations with >=2 each.

```bash
python scripts/rank_coordination_tasks.py PATH_TO_COORDINATION > candidates.json
```

The report ranks observed style divergence and robot decisions with at least two
unheld unfinished objects. It includes exact steps, delay until a subsequent robot
choice, both-agent failures, overlap, success, truncation and execution errors.
A delayed choice may happen after the relevant opportunity disappeared. Counts
are trajectory-dependent; this is a manual-review shortlist, NOT a demonstration
that the optimal robot assignment depends on partner identity. Inspect the first
few candidate timelines, then compare alternative robot choices in simulation
if needed. Such counterfactual branch rollouts are not implemented here.

Search does not use test traces as partner history. Runtime trial errors are
recorded and scanning continues after the next reset; setup/initialization errors
still stop the run. Results from failed or capped rollouts must not be interpreted
as task unsolvability. No automatic winner is declared when both profiles have
not produced error-free rows. Existing work-allocation runs are unchanged.
