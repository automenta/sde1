import unittest

from sde.core.types import Experiment, Trial
from sde.core.types import TrialStatus
from sde.core.types import WorkUnit
from sde.core.types import WorkUnitType
from sde.exploration.schedulers import SuccessiveHalvingScheduler


class TestSuccessiveHalvingScheduler(unittest.TestCase):

    def setUp(self):
        """Set up trials for testing the scheduler."""
        self.experiment = Experiment(id="test_exp")
        self.experiment.trials = {
            'trial_1': Trial(id='trial_1', algorithm_name='Algo1', hyperparameters={}),
            'trial_2': Trial(id='trial_2', algorithm_name='Algo2', hyperparameters={}),
            'trial_3': Trial(id='trial_3', algorithm_name='Algo3', hyperparameters={}),
            'trial_4': Trial(id='trial_4', algorithm_name='Algo4', hyperparameters={}),
        }
        # Keep a reference to the trials dict for convenience in other tests
        self.trials = self.experiment.trials

    def test_initial_work_units(self):
        """Test that the initial work units are generated correctly."""
        scheduler = SuccessiveHalvingScheduler(metric='accuracy', increasing=True)
        work_units = scheduler.get_initial_work_units(self.experiment)

        self.assertEqual(len(work_units), 4)
        self.assertTrue(all(isinstance(wu, WorkUnit) for wu in work_units))
        self.assertTrue(all(wu.type == WorkUnitType.TRAIN_EPOCH for wu in work_units))
        self.assertTrue(all(t.status == TrialStatus.ACTIVE for t in self.experiment.trials.values()))

    def test_pruning_logic_increasing_metric(self):
        """Test the pruning logic with a metric where higher is better."""
        scheduler = SuccessiveHalvingScheduler(metric='accuracy', increasing=True, min_epochs_per_rung=1, reduction_factor=2)

        # Simulate completion of the first rung
        for i, trial in enumerate(self.trials.values()):
            trial.status = TrialStatus.ACTIVE
            trial.current_epoch = 1
            trial.results['accuracy'] = [(1, 0.5 + i * 0.1)] # trial_4 is best

        # The trial that just finished is trial_1, but all have finished the rung
        finished_trial = self.trials['trial_1']
        new_work_units = scheduler.get_next_work_units(finished_trial, self.trials)

        # It should prune half the trials
        pruned_trials = [t for t in self.trials.values() if t.status == TrialStatus.PRUNED]
        active_trials = [t for t in self.trials.values() if t.status == TrialStatus.ACTIVE]

        self.assertEqual(len(pruned_trials), 2)
        self.assertEqual(len(active_trials), 2)
        self.assertEqual(len(new_work_units), 2)

        # Check that the worst-performing trials were pruned (trial_1 and trial_2)
        self.assertIn(self.trials['trial_1'], pruned_trials)
        self.assertIn(self.trials['trial_2'], pruned_trials)
        self.assertIn(self.trials['trial_3'], active_trials)
        self.assertIn(self.trials['trial_4'], active_trials)

    def test_pruning_logic_decreasing_metric(self):
        """Test the pruning logic with a metric where lower is better."""
        scheduler = SuccessiveHalvingScheduler(metric='loss', increasing=False, min_epochs_per_rung=1, reduction_factor=2)

        # Simulate completion of the first rung
        for i, trial in enumerate(self.trials.values()):
            trial.status = TrialStatus.ACTIVE
            trial.current_epoch = 1
            trial.results['loss'] = [(1, 0.5 - i * 0.1)] # trial_1 and trial_2 have highest loss

        finished_trial = self.trials['trial_1']
        new_work_units = scheduler.get_next_work_units(finished_trial, self.trials)

        pruned_trials = [t for t in self.trials.values() if t.status == TrialStatus.PRUNED]
        active_trials = [t for t in self.trials.values() if t.status == TrialStatus.ACTIVE]

        self.assertEqual(len(pruned_trials), 2)
        self.assertEqual(len(active_trials), 2)

        # Check that the worst-performing trials were pruned (trial_1 and trial_2 have the highest loss)
        self.assertIn(self.trials['trial_1'], pruned_trials)
        self.assertIn(self.trials['trial_2'], pruned_trials)
        self.assertIn(self.trials['trial_3'], active_trials)
        self.assertIn(self.trials['trial_4'], active_trials)

    def test_pruning_with_missing_metric(self):
        """Test that a trial missing a metric at a rung is correctly pruned."""
        scheduler = SuccessiveHalvingScheduler(metric='accuracy', increasing=True, min_epochs_per_rung=1, reduction_factor=2)

        # Simulate completion of the first rung for most trials
        self.trials['trial_1'].status = TrialStatus.ACTIVE
        self.trials['trial_1'].current_epoch = 1
        # trial_2 is missing its result
        self.trials['trial_2'].status = TrialStatus.ACTIVE
        self.trials['trial_2'].current_epoch = 1
        self.trials['trial_3'].status = TrialStatus.ACTIVE
        self.trials['trial_3'].current_epoch = 1
        self.trials['trial_3'].results['accuracy'] = [(1, 0.8)]
        self.trials['trial_4'].status = TrialStatus.ACTIVE
        self.trials['trial_4'].current_epoch = 1
        self.trials['trial_4'].results['accuracy'] = [(1, 0.9)]

        # The trial that just finished is trial_1
        finished_trial = self.trials['trial_1']
        scheduler.get_next_work_units(finished_trial, self.trials)

        # trial_1 and trial_2 should be pruned (trial_2 for missing metric, trial_1 for no metric)
        self.assertEqual(self.trials['trial_1'].status, TrialStatus.PRUNED)
        self.assertEqual(self.trials['trial_2'].status, TrialStatus.PRUNED)
        self.assertEqual(self.trials['trial_3'].status, TrialStatus.ACTIVE)
        self.assertEqual(self.trials['trial_4'].status, TrialStatus.ACTIVE)


if __name__ == '__main__':
    unittest.main()
