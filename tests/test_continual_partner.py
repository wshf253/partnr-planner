import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]

class ContinualTests(unittest.TestCase):
    def test_probe_feedback_never_updates_memory(self):
        core = types.ModuleType('habitat_llm.coordination.core')
        core.frozen_context = lambda condition, traces, *args, **kwargs: condition + json.dumps(traces)
        spec = importlib.util.spec_from_file_location('habitat_llm.coordination.continual', ROOT / 'habitat_llm/coordination/continual.py')
        m = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'habitat_llm.coordination.core': core}):
            spec.loader.exec_module(m)
        cfg = types.SimpleNamespace(seeds=[1], partners=[types.SimpleNamespace(id='p', profile='batch')], max_history_chars=18000)
        config = types.SimpleNamespace(coordination=cfg, continual=types.SimpleNamespace(rounds=3))
        calls, prompts = [], []
        def generate(prompt, **kwargs):
            prompts.append(prompt)
            return 'memory' + str(len(prompts))
        runner = types.SimpleNamespace(planner={0: types.SimpleNamespace(llm=types.SimpleNamespace(generate=generate))})
        def trial(runner, env, ep, profile, context, cfg, seed, directory):
            directory.mkdir()
            calls.append((ep['episode_id'], context))
            return {'episode_id': ep['episode_id']}, {'initial_observation_sha256': 'same'}
        with tempfile.TemporaryDirectory() as tmp, patch('builtins.print'):
            m.run_sequence(runner, None, [{'episode_id':'SUPPORT'}], [{'episode_id':'PROBE'}], config, Path(tmp), trial)
        self.assertEqual(len(calls), 14)
        self.assertEqual(len(prompts), 2)
        self.assertTrue(all('SUPPORT' in p and 'PROBE' not in p for p in prompts))
        self.assertIn('memory2', calls[11][1])
        self.assertIn('memory1', calls[12][1])
