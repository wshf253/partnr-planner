"""Four isolated probes per candidate; failures remain visible, no memory updates."""
import json
from .core import frozen_context


def select_candidates(candidates, ids):
    requested = [str(x) for x in ids]
    if not requested or len(requested) != len(set(requested)):
        raise ValueError('Need nonempty unique candidate_sweep.episode_ids')
    indexed = {str(ep['episode_id']): ep for ep in candidates}
    missing = set(requested) - set(indexed)
    if missing:
        raise ValueError('Missing or incompatible candidate episodes: ' + ', '.join(sorted(missing)))
    return [indexed[key] for key in requested]


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row['episode_id'], row['seed']), []).append(row)
    output = []
    for (episode, seed), trials in groups.items():
        comparisons = []
        for profile in ('batch', 'nearest'):
            pair = {r['condition']: r for r in trials if r['profile'] == profile}
            a, b = pair.get('no_history', {}), pair.get('oracle_style', {})
            valid = all(r.get('initial_state_matches') and not r.get('error') and
                        r.get('task_state_success') == 1 and r.get('stop_reason') == 'task_success'
                        for r in (a, b))
            comparisons.append(dict(profile=profile, comparable_successful_pair=valid,
                                    steps_saved_by_oracle=a['sim_steps'] - b['sim_steps'] if valid else None,
                                    robot_actions_changed=(a.get('robot_actions') != b.get('robot_actions')) if valid else None))
        baseline = [r for r in trials if r['condition'] == 'no_history']
        observed_difference = None
        if len(baseline) == 2 and all(r.get('initial_state_matches') and not r.get('error') and
                                                   r.get('task_state_success') == 1 for r in baseline):
            observed_difference = baseline[0]['public_trace_sha256'] != baseline[1]['public_trace_sha256']
        output.append(dict(episode_id=episode, seed=seed,
                           public_traces_differ=observed_difference,
                           comparisons=comparisons, trials=trials))
    return {'note': 'Shortlist only. Public trace differences do not prove identifiable traits; '
                    'oracle is not an optimal policy. Inspect action timing and failure logs. '
                    'Truncated/error runs are retained but excluded from success-time comparisons.',
            'candidates': output}


def run_sweep(runner, tests, config, out, trial):
    import hashlib
    rows = []
    cfg = config.coordination
    for seed in cfg.seeds:
        for ep in tests:
            for partner in cfg.partners:
                for condition in ('no_history', 'oracle_style'):
                    name = f'sweep_{condition}_{partner.id}_ep{ep["episode_id"]}_seed{seed}'
                    directory = out / name
                    row = dict(episode_id=str(ep['episode_id']), seed=int(seed),
                               partner_id=partner.id, profile=partner.profile,
                               condition=condition, log_directory=str(directory))
                    try:
                        trace, metrics = trial(runner, None, ep, partner.profile,
                                               frozen_context(condition, [], partner.profile),
                                               cfg, int(seed), directory)
                        row.update(metrics)
                        row['initial_state_matches'] = json.loads((directory / 'initial_state_check.json').read_text())['matches']
                        row['public_trace_sha256'] = hashlib.sha256(json.dumps(trace, sort_keys=True).encode()).hexdigest()
                        decisions = [json.loads(l) for l in (directory / 'robot_decisions.jsonl').read_text().splitlines()]
                        row['robot_actions'] = [d['action'] for d in decisions]
                    except Exception as exc:
                        row['error'] = f'{type(exc).__name__}: {exc}'
                    rows.append(row)
                    with (out / 'results.jsonl').open('a') as stream:
                        stream.write(json.dumps(row) + '\n')
                    (out / 'candidate_comparison.json').write_text(json.dumps(summarize(rows), indent=2))
                    print(json.dumps(row), flush=True)
