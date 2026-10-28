import unittest
from habitat_llm.coordination.search import select_search_candidates
from test_coordination_core import episode


class SearchTests(unittest.TestCase):
    def test_distinct_deterministic_and_input_preserved(self):
        episodes = [episode('a', 0), episode('b', 0), episode('c', 1)]
        selected = select_search_candidates(episodes, 10, 42)
        self.assertEqual(len(selected), 2)
        self.assertEqual(selected, select_search_candidates(list(reversed(episodes)), 10, 42))
        self.assertEqual(len(episodes), 3)
        self.assertEqual(len(select_search_candidates(episodes, 1, 42)), 1)

    def test_no_candidates_and_bad_limit(self):
        for candidates, limit in [([], 10), ([episode('a')], 0)]:
            with self.assertRaises(ValueError):
                select_search_candidates(candidates, limit, 42)
