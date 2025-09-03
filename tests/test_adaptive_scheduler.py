import unittest
from sde.exploration.schedulers import SuccessiveHalvingScheduler
from sde.core.types import Trial, TrialStatus, WorkUnit, WorkUnitType
from sde.engine.datastore import DataStore

class TestAsyncSuccessiveHalvingScheduler(unittest.TestCase):

    def setUp(self):
        """Set up trials and datastore for testing the new async scheduler."""
        self.trials = [
            Trial(id='trial_1', algorithm_name='Algo1', hyperparameters={}),
            Trial(id='trial_2', algorithm_name='Algo2', hyperparameters={}),
            Trial(id='trial_3', algorithm_name='Algo3', hyperparameters={}),
            Trial(id='trial_4', algorithm_name='Algo4', hyperparameters={}),
        ]
        self.datastore = DataStore(self.trials)
        self.metric = 'accuracy'
        self.scheduler = SuccessiveHalvingScheduler(
            metric=self.metric,
            increasing=True,
            min_epochs_per_rung=2, # Rungs at epoch 2, 4, 8...
            reduction_factor=2
        )
        # Activate all trials for tests
        for t in self.trials:
            self.datastore.set_trial_status(t.id, TrialStatus.ACTIVE)

    def test_initial_work_units(self):
        """Test that the initial work units are generated correctly."""
        # Reset trials to PENDING
        for t in self.trials:
            self.datastore.set_trial_status(t.id, TrialStatus.PENDING)

        work_units = self.scheduler.get_initial_work_units(self.datastore)

        self.assertEqual(len(work_units), 4)
        self.assertTrue(all(isinstance(wu, WorkUnit) for wu in work_units))
        self.assertTrue(all(self.datastore.get_trial(wu.trial_id).status == TrialStatus.ACTIVE for wu in work_units))

    def test_trial_continues_before_first_rung(self):
        """A trial that has not reached the first rung should continue."""
        trial1 = self.datastore.get_trial('trial_1')
        trial1.current_epoch = 1 # First rung is at epoch 2
        trial1.results[self.metric] = [(1, 0.5)]

        work_units = self.scheduler.get_next_work_units(trial1, self.datastore)

        self.assertEqual(len(work_units), 1)
        self.assertEqual(work_units[0].trial_id, 'trial_1')
        self.assertEqual(trial1.status, TrialStatus.ACTIVE)

    def test_pruning_decision_deferred_if_not_enough_contemporaries(self):
        """A trial at a rung should continue if not enough others are there to be compared against."""
        # trial_1 reaches the first rung at epoch 2
        trial1 = self.datastore.get_trial('trial_1')
        trial1.current_epoch = 2
        trial1.results[self.metric] = [(1, 0.5), (2, 0.6)]

        # No other trials have reached epoch 2, so no decision can be made.
        work_units = self.scheduler.get_next_work_units(trial1, self.datastore)

        self.assertEqual(len(work_units), 1)
        self.assertEqual(work_units[0].trial_id, 'trial_1')
        self.assertEqual(trial1.status, TrialStatus.ACTIVE)

    def test_trial_is_pruned_at_rung(self):
        """A trial should be pruned if it's the worst performer at a rung."""
        # All 4 trials reach epoch 2. trial_1 is the worst.
        # reduction_factor=2 means 4/2=2 trials should be pruned.
        t1 = self.datastore.get_trial('trial_1'); t1.current_epoch = 2; t1.results[self.metric] = [(1, 0.1), (2, 0.1)]
        t2 = self.datastore.get_trial('trial_2'); t2.current_epoch = 2; t2.results[self.metric] = [(1, 0.2), (2, 0.2)]
        t3 = self.datastore.get_trial('trial_3'); t3.current_epoch = 2; t3.results[self.metric] = [(1, 0.3), (2, 0.3)]
        t4 = self.datastore.get_trial('trial_4'); t4.current_epoch = 2; t4.results[self.metric] = [(1, 0.4), (2, 0.4)]

        # The scheduler is called for trial_1, which should be pruned.
        work_units = self.scheduler.get_next_work_units(t1, self.datastore)

        self.assertEqual(len(work_units), 0) # No new work
        self.assertEqual(t1.status, TrialStatus.PRUNED)

    def test_trial_survives_rung(self):
        """A trial should continue if it's a top performer at a rung."""
        # All 4 trials reach epoch 2. trial_4 is the best.
        t1 = self.datastore.get_trial('trial_1'); t1.current_epoch = 2; t1.results[self.metric] = [(1, 0.1), (2, 0.1)]
        t2 = self.datastore.get_trial('trial_2'); t2.current_epoch = 2; t2.results[self.metric] = [(1, 0.2), (2, 0.2)]
        t3 = self.datastore.get_trial('trial_3'); t3.current_epoch = 2; t3.results[self.metric] = [(1, 0.3), (2, 0.3)]
        t4 = self.datastore.get_trial('trial_4'); t4.current_epoch = 2; t4.results[self.metric] = [(1, 0.4), (2, 0.4)]

        # The scheduler is called for trial_4, which should survive.
        work_units = self.scheduler.get_next_work_units(t4, self.datastore)

        self.assertEqual(len(work_units), 1)
        self.assertEqual(work_units[0].trial_id, 'trial_4')
        self.assertEqual(t4.status, TrialStatus.ACTIVE)

    def test_decreasing_metric_pruning(self):
        """Test pruning works correctly for a decreasing metric like 'loss'."""
        self.scheduler = SuccessiveHalvingScheduler(metric='loss', increasing=False, min_epochs_per_rung=2, reduction_factor=2)

        # Lower loss is better. trial_4 now has the worst performance.
        t1 = self.datastore.get_trial('trial_1'); t1.current_epoch = 2; t1.results['loss'] = [(1, 0.1), (2, 0.1)]
        t2 = self.datastore.get_trial('trial_2'); t2.current_epoch = 2; t2.results['loss'] = [(1, 0.2), (2, 0.2)]
        t3 = self.datastore.get_trial('trial_3'); t3.current_epoch = 2; t3.results['loss'] = [(1, 0.3), (2, 0.3)]
        t4 = self.datastore.get_trial('trial_4'); t4.current_epoch = 2; t4.results['loss'] = [(1, 0.4), (2, 0.4)]

        # When the scheduler is called for trial_4, it should be pruned.
        work_units = self.scheduler.get_next_work_units(t4, self.datastore)
        self.assertEqual(len(work_units), 0)
        self.assertEqual(t4.status, TrialStatus.PRUNED)

        # When the scheduler is called for trial_1, it should survive.
        work_units = self.scheduler.get_next_work_units(t1, self.datastore)
        self.assertEqual(len(work_units), 1)
        self.assertEqual(work_units[0].trial_id, 'trial_1')
        self.assertEqual(t1.status, TrialStatus.ACTIVE)

if __name__ == '__main__':
    unittest.main()
