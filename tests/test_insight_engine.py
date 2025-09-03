import unittest
from sde.engine.insight import InsightEngine
from sde.core.types import Trial, TrialStatus
from sde.engine.datastore import DataStore

class TestInsightEngine(unittest.TestCase):

    def setUp(self):
        self.trials = [
            Trial(id='t1', algorithm_name='A1', hyperparameters={}, status=TrialStatus.ACTIVE),
            Trial(id='t2', algorithm_name='A2', hyperparameters={}, status=TrialStatus.ACTIVE),
            Trial(id='t3', algorithm_name='A3', hyperparameters={}, status=TrialStatus.ACTIVE),
        ]
        self.datastore = DataStore(self.trials)
        self.metric = 'accuracy'
        self.engine = InsightEngine(self.datastore, self.metric, higher_is_better=True)

    def test_best_performer_insight(self):
        """Test that the best performer insight is correctly identified."""
        # t1 gets a good score
        t1 = self.datastore.get_trial('t1')
        t1.results[self.metric] = [(1, 0.8)]
        insights = self.engine.analyze(t1)
        self.assertEqual(len(insights), 1)
        self.assertEqual(insights[0].type, "BEST_PERFORMER")

        # t2 gets a better score
        t2 = self.datastore.get_trial('t2')
        t2.results[self.metric] = [(1, 0.9)]
        insights = self.engine.analyze(t2)
        self.assertEqual(len(insights), 1)
        self.assertEqual(insights[0].type, "BEST_PERFORMER")
        self.assertIn("t2", insights[0].message)

        # t3 gets a score that is not the best
        t3 = self.datastore.get_trial('t3')
        t3.results[self.metric] = [(1, 0.85)]
        insights = self.engine.analyze(t3)
        self.assertEqual(len(insights), 0) # No new best performer

        # t1 improves but is still not the best
        t1.results[self.metric].append((2, 0.88))
        insights = self.engine.analyze(t1)
        self.assertEqual(len(insights), 0)

    def test_plateau_insight_fires_once(self):
        """Test that the plateau insight fires only once for a trial."""
        t1 = self.datastore.get_trial('t1')
        t2 = self.datastore.get_trial('t2')

        # First, establish a clear best performer that is not t1.
        t2.results[self.metric] = [(1, 0.95)]
        self.engine.analyze(t2) # This sets t2 as the best performer.

        # Now, test the plateau logic on t1 in isolation.
        # This history should trigger the plateau insight.
        t1.results[self.metric] = [(1, 0.8), (2, 0.801), (3, 0.802), (4, 0.803)]
        t1.current_epoch = 4

        insights = self.engine.analyze(t1)
        # We expect exactly one insight, and it should be for the plateau.
        self.assertEqual(len(insights), 1)
        self.assertEqual(insights[0].type, "PLATEAU")
        self.assertTrue(self.engine._has_fired(t1.id, "PLATEAU"))

        # Now, advance t1. It's still plateauing, but the insight should not fire again.
        t1.results[self.metric].append((5, 0.804))
        t1.current_epoch = 5
        insights = self.engine.analyze(t1)
        self.assertEqual(len(insights), 0, "A plateau insight should not fire twice for the same trial.")

    def test_fired_insights_are_tracked(self):
        """Test the internal tracking of fired insights."""
        t1 = self.datastore.get_trial('t1')
        t1.current_epoch = 1
        t1.results[self.metric] = [(1, 0.1)]

        # Fire a poor performance insight
        # (mocking other trials for z-score calculation)
        t2 = self.datastore.get_trial('t2'); t2.current_epoch=1; t2.results[self.metric]=[(1, 0.9)]
        t3 = self.datastore.get_trial('t3'); t3.current_epoch=1; t3.results[self.metric]=[(1, 0.95)]

        insights = self.engine.analyze(t1)
        self.assertEqual(len(insights), 1)
        self.assertEqual(insights[0].type, "POOR_INITIAL_PERFORMANCE")
        self.assertTrue(self.engine._has_fired('t1', "POOR_INITIAL_PERFORMANCE"))

        # Try to analyze again, should not produce the same insight
        insights = self.engine.analyze(t1)
        self.assertEqual(len(insights), 0)


if __name__ == '__main__':
    unittest.main()
