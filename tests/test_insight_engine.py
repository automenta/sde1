import unittest

from sde.core.domain import Trial
from sde.core.domain import TrialStatus
from sde.engine.insight import InsightEngine


class TestInsightEngine(unittest.TestCase):
    def setUp(self):
        """Set up a basic InsightEngine and mock trials for testing."""
        self.trials = {}
        self.primary_metric = "accuracy"
        self.higher_is_better = True
        self.engine = InsightEngine(
            self.trials, self.primary_metric, self.higher_is_better
        )

    def _create_trial(
        self, trial_id, status=TrialStatus.ACTIVE, results=None, hparams=None
    ):
        """Helper to create a trial and add it to the engine's trial dict."""
        if results is None:
            results = {}
        if hparams is None:
            hparams = {}
        trial = Trial(
            id=trial_id,
            algorithm_name=f"Algo_{trial_id}",
            status=status,
            results={self.primary_metric: results},
            hyperparameters=hparams,
            current_epoch=len(results),
        )
        self.trials[trial_id] = trial
        return trial

    def test_detect_best_performer(self):
        """Test the _detect_best_performer method."""
        trial1 = self._create_trial("t1", results=[(1, 0.8)])
        insight = self.engine._detect_best_performer(trial1)
        self.assertIsNotNone(insight)
        self.assertEqual(insight.type, "BEST_PERFORMER")
        self.assertEqual(self.engine.current_best_trial_id, "t1")

        # Add a second, better trial
        trial2 = self._create_trial("t2", results=[(1, 0.9)])
        insight = self.engine._detect_best_performer(trial2)
        self.assertIsNotNone(insight)
        self.assertEqual(insight.type, "BEST_PERFORMER")
        self.assertEqual(self.engine.current_best_trial_id, "t2")

        # Add a third, worse trial
        trial3 = self._create_trial("t3", results=[(1, 0.85)])
        insight = self.engine._detect_best_performer(trial3)
        self.assertIsNone(insight)
        self.assertEqual(self.engine.current_best_trial_id, "t2")

    def test_detect_performance_plateau(self):
        """Test the _detect_performance_plateau method."""
        # Plateaued trial
        trial1 = self._create_trial(
            "t1", results=[(1, 0.8), (2, 0.801), (3, 0.802), (4, 0.803)]
        )
        insight = self.engine._detect_performance_plateau(trial1)
        self.assertIsNotNone(insight)
        self.assertEqual(insight.type, "PLATEAU")

        # Improving trial
        trial2 = self._create_trial(
            "t2", results=[(1, 0.8), (2, 0.85), (3, 0.9), (4, 0.95)]
        )
        insight = self.engine._detect_performance_plateau(trial2)
        self.assertIsNone(insight)

        # Test that insight is only fired once
        self.engine._fired_plateau_insights.add("t1")
        insight = self.engine._detect_performance_plateau(trial1)
        self.assertIsNone(insight)

    def test_detect_poor_initial_performance(self):
        """Test the _detect_poor_initial_performance method."""
        # Create a baseline of peers
        self._create_trial("peer1", results=[(1, 0.8)])
        self._create_trial("peer2", results=[(1, 0.82)])
        self._create_trial("peer3", results=[(1, 0.78)])

        # Create a trial that is a clear outlier
        outlier_trial = self._create_trial("outlier", results=[(1, 0.5)])
        insight = self.engine._detect_poor_initial_performance(outlier_trial)
        self.assertIsNotNone(insight)
        self.assertEqual(insight.type, "POOR_INITIAL_PERFORMANCE")

        # Create a trial that is not an outlier
        normal_trial = self._create_trial("normal", results=[(1, 0.79)])
        insight = self.engine._detect_poor_initial_performance(normal_trial)
        self.assertIsNone(insight)

    def test_detect_performance_crossover(self):
        """Test the _detect_performance_crossover method."""
        trial1 = self._create_trial("t1", results=[(1, 0.8), (2, 0.85)])  # Winner
        self._create_trial("t2", results=[(1, 0.82), (2, 0.83)])  # Loser

        # Update trial1 to its new state (epoch 2)
        trial1.results[self.primary_metric] = [(1, 0.8), (2, 0.85)]
        trial1.current_epoch = 2

        insights = self.engine._detect_performance_crossover(trial1)
        self.assertEqual(len(insights), 1)
        insight = insights[0]
        self.assertEqual(insight.type, "PERFORMANCE_CROSSOVER")
        self.assertIn("t1", insight.trial_ids)
        self.assertIn("t2", insight.trial_ids)

        # Check that it doesn't fire again for the same crossover
        insights = self.engine._detect_performance_crossover(trial1)
        self.assertEqual(len(insights), 0)

    def test_detect_hyperparameter_correlation_categorical(self):
        """Test h-param correlation with a categorical parameter."""
        # Group 1: Adam optimizer (good performance)
        self._create_trial(
            "t1",
            status=TrialStatus.COMPLETED,
            results=[(1, 0.90)],
            hparams={"optimizer_params": {"name": "Adam"}},
        )
        self._create_trial(
            "t2",
            status=TrialStatus.COMPLETED,
            results=[(1, 0.92)],
            hparams={"optimizer_params": {"name": "Adam"}},
        )
        self._create_trial(
            "t3",
            status=TrialStatus.COMPLETED,
            results=[(1, 0.91)],
            hparams={"optimizer_params": {"name": "Adam"}},
        )
        # Group 2: SGD optimizer (poor performance)
        self._create_trial(
            "t4",
            status=TrialStatus.COMPLETED,
            results=[(1, 0.70)],
            hparams={"optimizer_params": {"name": "SGD"}},
        )
        self._create_trial(
            "t5",
            status=TrialStatus.COMPLETED,
            results=[(1, 0.72)],
            hparams={"optimizer_params": {"name": "SGD"}},
        )
        self._create_trial(
            "t6",
            status=TrialStatus.COMPLETED,
            results=[(1, 0.71)],
            hparams={"optimizer_params": {"name": "SGD"}},
        )

        insights = self.engine._detect_hyperparameter_correlation(self.trials["t6"])
        self.assertEqual(len(insights), 1)
        insight = insights[0]
        self.assertEqual(insight.type, "HYPERPARAM_CORRELATION")
        # Make assertions less brittle
        self.assertIn("'optimizer_params.name'", insight.message)
        self.assertIn("group 'Adam'", insight.message)
        self.assertIn("outperforms group 'SGD'", insight.message)

    def test_detect_hyperparameter_correlation_numerical(self):
        """Test h-param correlation with a numerical parameter (learning rate)."""
        # To meet the min_group_size of 3, we need more data points
        # Group 1: Low LR (good performance)
        self._create_trial(
            "t1_low",
            status=TrialStatus.COMPLETED,
            results=[(1, 0.95)],
            hparams={"optimizer_params": {"lr": 0.001}},
        )
        self._create_trial(
            "t2_low",
            status=TrialStatus.COMPLETED,
            results=[(1, 0.96)],
            hparams={"optimizer_params": {"lr": 0.002}},
        )
        self._create_trial(
            "t3_low",
            status=TrialStatus.COMPLETED,
            results=[(1, 0.94)],
            hparams={"optimizer_params": {"lr": 0.0015}},
        )
        # Group 2: High LR (poor performance)
        self._create_trial(
            "t4_high",
            status=TrialStatus.COMPLETED,
            results=[(1, 0.60)],
            hparams={"optimizer_params": {"lr": 0.1}},
        )
        self._create_trial(
            "t5_high",
            status=TrialStatus.COMPLETED,
            results=[(1, 0.62)],
            hparams={"optimizer_params": {"lr": 0.2}},
        )
        self._create_trial(
            "t6_high",
            status=TrialStatus.COMPLETED,
            results=[(1, 0.61)],
            hparams={"optimizer_params": {"lr": 0.15}},
        )
        # Group 3: Medium LR (medium performance) - to ensure binning works
        self._create_trial(
            "t7_med",
            status=TrialStatus.COMPLETED,
            results=[(1, 0.80)],
            hparams={"optimizer_params": {"lr": 0.01}},
        )
        self._create_trial(
            "t8_med",
            status=TrialStatus.COMPLETED,
            results=[(1, 0.82)],
            hparams={"optimizer_params": {"lr": 0.02}},
        )
        self._create_trial(
            "t9_med",
            status=TrialStatus.COMPLETED,
            results=[(1, 0.81)],
            hparams={"optimizer_params": {"lr": 0.015}},
        )

        insights = self.engine._detect_hyperparameter_correlation(self.trials["t9_med"])
        self.assertEqual(len(insights), 1)
        insight = insights[0]
        self.assertEqual(insight.type, "HYPERPARAM_CORRELATION")
        self.assertIn("'optimizer_params.lr'", insight.message)
        # Check that 'low' outperforms 'high'
        self.assertIn("group 'low'", insight.message)
        self.assertIn("outperforms group 'high'", insight.message)


if __name__ == "__main__":
    unittest.main()
