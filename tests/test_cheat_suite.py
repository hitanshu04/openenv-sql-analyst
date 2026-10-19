"""
The cheat suite as a test: every scripted agent on every task must obey its
category's rule (see adversaries/agents.py for the threat model).

Run:  python tests/test_cheat_suite.py      (scoreboard: python -m adversaries)
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adversaries.suite import run_suite


class TestCheatSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.verdicts = run_suite()

    def _check(self, category):
        checked = [v for v in self.verdicts if v.category == category]
        self.assertTrue(checked, f"no {category} agents ran")
        for v in checked:
            with self.subTest(task=v.episode.task_id, agent=v.episode.agent):
                self.assertIsNone(v.problem, v.problem)

    def test_honest_agents_succeed_and_efficiency_ranks_first(self):
        self._check("honest")

    def test_phrasing_of_a_correct_answer_does_not_matter(self):
        self._check("variant")

    def test_no_process_cheat_beats_an_honest_agent(self):
        self._check("process")

    def test_no_knowledge_cheat_is_graded_correct(self):
        self._check("knowledge")

    def test_vandals_are_refused_and_change_nothing(self):
        self._check("vandal")


if __name__ == "__main__":
    unittest.main(verbosity=2)
