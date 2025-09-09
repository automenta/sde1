import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

from sde.core.types import Experiment, Trial
from sde.core.types import ExecutionSettings
from sde.core.types import TrialStatus
from sde.core.types import WorkUnit
from sde.core.types import WorkUnitType
from sde.engine.runtime import SdeRuntimeEngine


class TestSdeRuntimeEngine(unittest.TestCase):

    @patch('sde.engine.runtime.ComputeScheduler')
    @patch('sde.engine.runtime.SchedulerFactory')
    def setUp(self, MockSchedulerFactory, MockComputeScheduler):
        """Set up a runtime engine instance for tests."""
        self.mock_event_callback = MagicMock()
        self.runtime_engine = SdeRuntimeEngine(event_callback=self.mock_event_callback)

        # Mock the components that the engine creates internally
        self.mock_adaptive_scheduler = MockSchedulerFactory.create_scheduler.return_value
        self.mock_adaptive_scheduler.get_initial_work_units.return_value = ([], {})
        self.mock_compute_scheduler = MockComputeScheduler.return_value

        # Define a basic experiment and trial
        self.experiment = Experiment(id="test_exp_runtime")
        self.experiment.challenge = {"name": "MNIST", "type": "vision"}
        self.trial = Trial(id='trial1', algorithm_name='TestAlgo', hyperparameters={'lr': 0.1})
        self.experiment.trials = {'trial1': self.trial}
        self.experiment.execution_settings = ExecutionSettings(
            num_workers=1,
            num_trials_per_algo=1,
            enable_checkpointing=False,
            work_unit_timeout_seconds=300,
        )

        # Initialize the engine's internal state by calling the start handler directly
        self.runtime_engine._handle_start_run({
            "experiment_definition": self.experiment.to_dict()
        })

    def test_work_unit_error_sets_trial_to_failed(self):
        """Test that if a work unit result contains an error, the trial's status
        is set to FAILED and a TRIAL_UPDATED event is emitted.
        """
        work_unit = WorkUnit(trial_id='trial1', type=WorkUnitType.TRAIN_EPOCH, payload={})
        error_result = {"error": "CUDA out of memory"}

        # Directly call the method that processes results
        self.runtime_engine._process_completed_work_unit(work_unit, error_result)

        # 1. Check that the datastore has the updated trial status
        updated_trial = self.runtime_engine.datastore.get_trial('trial1')
        self.assertEqual(updated_trial.status, TrialStatus.FAILED)

        # 2. Check that a TRIAL_UPDATED event was emitted with the failure
        self.mock_event_callback.assert_called_with(
            "TRIAL_UPDATED",
            {"trial": updated_trial.to_dict()}
        )

        # 3. Check that no new work was scheduled
        self.mock_adaptive_scheduler.get_next_work_units.assert_not_called()
