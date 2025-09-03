import unittest
from sde.exploration.schedulers import SuccessiveHalvingScheduler
from sde.core.types import Trial, TrialStatus, WorkUnit, WorkUnitType

class TestSuccessiveHalvingScheduler(unittest.TestCase):

    def setUp(self):
        """Set up trials for testing the scheduler."""
        self.trials = {
            'trial_1': Trial(id='trial_1', algorithm_name='Algo1', hyperparameters={}),
            'trial_2': Trial(id='trial_2', algorithm_name='Algo2', hyperparameters={}),
            'trial_3': Trial(id='trial_3', algorithm_name='Algo3', hyperparameters={}),
            'trial_4': Trial(id='trial_4', algorithm_name='Algo4', hyperparameters={}),
        }

    def test_initial_work_units(self):
        """Test that the initial work units are generated correctly."""
        scheduler = SuccessiveHalvingScheduler(metric='accuracy', increasing=True)
        work_units = scheduler.get_initial_work_units(self.trials)

        self.assertEqual(len(work_units), 4)
        self.assertTrue(all(isinstance(wu, WorkUnit) for wu in work_units))
        self.assertTrue(all(wu.type == WorkUnitType.TRAIN_EPOCH for wu in work_units))
        self.assertTrue(all(t.status == TrialStatus.ACTIVE for t in self.trials.values()))

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
            trial.results['loss'] = [(1, 0.5 - i * 0.1)] # trial_4 is worst

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


if __name__ == '__main__':
    unittest.main()
