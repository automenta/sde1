import unittest
from unittest.mock import MagicMock, patch

from sde.core.types import Trial, TrialStatus, WorkUnit, WorkUnitType
from sde.engine.runtime import SdeRuntimeEngine


class TestSdeRuntimeEngine(unittest.TestCase):

    def setUp(self):
        """Set up common mocks and a trial for tests."""
        self.mock_adaptive_scheduler = MagicMock()
        self.mock_trial_updated_callback = MagicMock()
        self.mock_insights_callback = MagicMock()
        self.trial = Trial(id='trial1', algorithm_name='TestAlgo', hyperparameters={'lr': 0.1})

        # We need to patch the ComputeScheduler as it tries to create a process pool
        with patch('sde.engine.runtime.ComputeScheduler') as MockComputeScheduler:
            self.runtime_engine = SdeRuntimeEngine(
                trials=[self.trial],
                dataset_name="MNIST",
                adaptive_scheduler=self.mock_adaptive_scheduler,
                trial_updated_callback=self.mock_trial_updated_callback,
                insights_callback=self.mock_insights_callback,
                max_workers=1,
            )

    def test_work_unit_error_sets_trial_to_failed(self):
        """
        Test that if a work unit result contains an error, the trial's status
        is set to FAILED and the UI callback is notified.
        """
        work_unit = WorkUnit(trial_id='trial1', type=WorkUnitType.TRAIN_EPOCH)
        error_result = {"error": "CUDA out of memory"}

        # Directly call the method that processes results
        self.runtime_engine._process_completed_work_unit(work_unit, error_result)

        # 1. Check that the datastore has the updated trial status
        # (We need to access the internal datastore for this check)
        updated_trial = self.runtime_engine.datastore.get_trial('trial1')
        self.assertEqual(updated_trial.status, TrialStatus.FAILED)

        # 2. Check that the UI callback was called with the failure
        self.mock_trial_updated_callback.assert_called_once()
        callback_args = self.mock_trial_updated_callback.call_args[0][0]
        self.assertEqual(callback_args['id'], 'trial1')
        self.assertEqual(callback_args['status'], 'FAILED')

        # 3. Check that no new work was scheduled
        self.mock_adaptive_scheduler.get_next_work_units.assert_not_called()
