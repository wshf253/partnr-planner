"""Accumulate partner memory on support; evaluate without test feedback leakage."""
import json
from .core import frozen_context


def update_memory(llm, previous, trace, directory, max_chars):
    evidence = frozen_context('history', [trace], max_chars=max_chars)
    prompt = (
        'Update a concise model of this same collaborative partner from observed evidence. '
        'Agent 0 is the robot (self); agent 1 is the partner. Do not attribute self actions '
        'to the partner. Separate observations from hypotheses. Include behavioral tendencies, '
        'evidence, uncertainty, counterexamples, and planning implications. Consider situational '
        'constraints before revising a trait. Do not memorize object IDs or prescribe exact task '
        'solutions. No hidden partner labels are available. Return plain text under 3000 characters.\n'
        'Previous memory:\n' + previous + '\nNew observed collaboration:\n' + evidence
    )
    (directory / 'memory_update_prompt.txt').write_text(prompt)
    updated = llm.generate(prompt, max_length=1024)
    (directory / 'memory_update_raw.txt').write_text(updated)
    if not updated.strip() or len(updated) > 3000:
        raise ValueError('Memory must be nonempty and at most 3000 characters')
    return updated


def run_sequence(runner, env, support, tests, config, out, run_trial):
    cfg = config.coordination
    rounds = int(config.continual.rounds)
    if rounds < 1 or not support or not tests:
        raise ValueError('Continual pilot needs support, tests, and positive rounds')
    for seed in cfg.seeds:
        for partner in cfg.partners:
            memory = ''
            first_memory = ''
            for day in range(1, rounds + 1):
                # Evaluate memory BEFORE this day's interaction; probes never update it.
                for ep in tests:
                    initial = None
                    for condition in ('no_history', 'memory', 'fixed_memory', 'oracle_style'):
                        selected = first_memory if condition == 'fixed_memory' else memory
                        context = (
                            'Inferred partner model from previous collaborations; uncertain, '
                            'not ground truth. Use current evidence to coordinate.\n' + selected
                            if condition in ('memory', 'fixed_memory') and selected
                            else frozen_context('oracle_style' if condition == 'oracle_style'
                                                else 'no_history', [], partner.profile)
                        )
                        name = f'day{day}_{condition}_{partner.id}_ep{ep["episode_id"]}_seed{seed}'
                        row = dict(day=day, phase='probe', condition=condition,
                                   partner_id=partner.id, profile=partner.profile,
                                   episode_id=str(ep['episode_id']), seed=int(seed))
                        try:
                            _, metrics = run_trial(runner, env, ep, partner.profile,
                                                   context, cfg, int(seed), out / name)
                            row.update(metrics)
                            if initial is None:
                                initial = row['initial_observation_sha256']
                            row['paired_initial_state_matches'] = initial == row['initial_observation_sha256']
                        except Exception as exc:
                            row['error'] = f'{type(exc).__name__}: {exc}'
                        with (out / 'results.jsonl').open('a') as f:
                            f.write(json.dumps(row) + '\n')
                        print(json.dumps(row), flush=True)
                if day == rounds:
                    continue
                ep = support[(day - 1) % len(support)]
                directory = out / f'day{day}_experience_{partner.id}_ep{ep["episode_id"]}_seed{seed}'
                context = ('Inferred partner model:\n' + memory if memory else
                           frozen_context('no_history', []))
                # One shared experience stream makes memory ablations comparable.
                # Abort on support/update failure rather than silently using stale memory.
                trace, metrics = run_trial(runner, env, ep, partner.profile, context,
                                           cfg, int(seed), directory)
                (directory / 'memory_before.txt').write_text(memory)
                (directory / 'experience_metrics.json').write_text(json.dumps(metrics, indent=2))
                memory = update_memory(runner.planner[0].llm, memory, trace, directory,
                                       int(cfg.max_history_chars))
                if not first_memory:
                    first_memory = memory
                (directory / 'memory_after.txt').write_text(memory)
