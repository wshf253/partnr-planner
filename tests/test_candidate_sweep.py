import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

class SweepTests(unittest.TestCase):
    def test_selection_and_failed_pairs(self):
        path = Path(__file__).resolve().parents[1] / 'habitat_llm/coordination/candidate_sweep.py'
        core = types.ModuleType('habitat_llm.coordination.core')
        core.frozen_context = lambda *a: ''
        spec = importlib.util.spec_from_file_location('habitat_llm.coordination.candidate_sweep', path)
        m = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'habitat_llm.coordination.core': core}):
            spec.loader.exec_module(m)
        self.assertEqual(m.select_candidates([{'episode_id':1},{'episode_id':2}], ['2','1'])[0]['episode_id'], 2)
        with self.assertRaises(ValueError):
            m.select_candidates([{'episode_id':1}], ['2'])
        a = dict(episode_id='1', seed=1, profile='batch', condition='no_history',
                 initial_state_matches=True, task_state_success=1, stop_reason='task_success',
                 sim_steps=100, robot_actions=['a'])
        b = dict(a, condition='oracle_style', sim_steps=80, robot_actions=['b'])
        def compare(b):
            return m.summarize([a,b])['candidates'][0]['comparisons'][0]
        self.assertEqual(compare(b)['steps_saved_by_oracle'], 20)
        self.assertTrue(compare(b)['robot_actions_changed'])
        for overrides in [dict(error='failed'), dict(initial_state_matches=False),
                          dict(task_state_success=0), dict(stop_reason='sim_step_limit')]:
            self.assertIsNone(compare(dict(b, **overrides))['steps_saved_by_oracle'])
