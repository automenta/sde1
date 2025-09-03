import unittest
from unittest.mock import MagicMock, patch, ANY

from sde.core.types import Trial, WorkUnit, WorkUnitType, TrialStatus
from sde.engine.orchestrator import Orchestrator
from sde.exploration.schedulers import AdaptiveScheduler

class TestOrchestrator(unittest.TestCase):

    @patch('sde.engine.orchestrator.InsightEngine', autospec=True)
    @patch('sde.engine.orchestrator.Scheduler', autospec=True)
    @patch('sde.engine.orchestrator.DataStore', autospec=True)
    def setUp(self, MockDataStore, MockScheduler, MockInsightEngine):
        """Set up a basic orchestrator with mock components for testing."""
        self.trials = [
            Trial(id='trial_1', algorithm_name='Algo1', hyperparameters={'lr': 0.1}),
            Trial(id='trial_2', algorithm_name='Algo2', hyperparameters={'lr': 0.01}),
        ]
        self.dataset_name = "TestDataset"
        self.adaptive_scheduler = MagicMock(spec=AdaptiveScheduler)
        self.adaptive_scheduler.metric = "accuracy"
        self.adaptive_scheduler.increasing = True

        # Store mock classes for assertion
        self.MockDataStore = MockDataStore
        self.MockScheduler = MockScheduler
        self.MockInsightEngine = MockInsightEngine

        # Instantiate the Orchestrator. The mocks will be automatically used.
        self.orchestrator = Orchestrator(
            trials=self.trials,
            dataset_name=self.dataset_name,
            adaptive_scheduler=self.adaptive_scheduler,
            max_workers=2
        )

        # Get references to the mock instances created inside the Orchestrator
        self.mock_datastore_instance = self.orchestrator.datastore
        self.mock_scheduler_instance = self.orchestrator.scheduler
        self.mock_insight_engine_instance = self.orchestrator.insight_engine

    def test_initialization(self):
        """Test that the orchestrator initializes its components correctly."""
        self.MockDataStore.assert_called_once_with(self.trials)
        # The datastore instance's get_all_trials is called by InsightEngine
        self.mock_datastore_instance.get_all_trials.assert_called_once()
        self.MockInsightEngine.assert_called_once_with(
            self.mock_datastore_instance.get_all_trials.return_value,
            primary_metric="accuracy",
            higher_is_better=True
        )
        self.MockScheduler.assert_called_once_with(
            datastore=self.mock_datastore_instance,
            dataset_name=self.dataset_name,
            max_workers=2,
            enable_checkpointing=False,
            checkpoints_dir='./checkpoints'
        )

    def test_run_experiment_end_to_end(self):
        """Test a full, simplified, successful run of an experiment."""
        # --- Arrange ---
        # 1. Profiling phase
        profiling_work = [WorkUnit(trial_id='trial_1', type=WorkUnitType.PROFILE_SPEED)]
        self.orchestrator._get_profiling_work = MagicMock(return_value=profiling_work)
        profiling_result = {'profile_results': {'est_time_per_epoch': 0.5}}

        # 2. Main experiment phase
        initial_work = [WorkUnit(trial_id='trial_1', type=WorkUnitType.TRAIN_EPOCH)]
        self.adaptive_scheduler.get_initial_work_units.return_value = initial_work
        training_result = {'metrics': {'accuracy': 0.9}, 'state_updates': {'current_epoch': 1}}

        # Mock the scheduler's run generator
        self.mock_scheduler_instance.run.side_effect = [
            iter([(profiling_work[0], profiling_result)]),
            iter([(initial_work[0], training_result)])
        ]

        # Mock datastore returns
        self.mock_datastore_instance.get_trial.return_value = self.trials[0]
        self.mock_datastore_instance.get_all_trials.return_value = {t.id: t for t in self.trials}

        # Mock adaptive scheduler to return no new work, ending the loop
        self.adaptive_scheduler.get_next_work_units.return_value = []

        # --- Act ---
        thread = self.orchestrator.start()
        thread.join(timeout=2)

        # --- Assert ---
        self.mock_scheduler_instance.start.assert_called_once()
        self.mock_datastore_instance.update_algorithm_profile.assert_called_once_with('Algo1', 0.5)
        self.adaptive_scheduler.get_initial_work_units.assert_called_once()
        self.mock_datastore_instance.update_trial_state.assert_called_once_with(initial_work[0], training_result)
        self.mock_insight_engine_instance.analyze.assert_called_once()
        self.adaptive_scheduler.get_next_work_units.assert_called_once()
        self.mock_scheduler_instance.stop.assert_called_once()
        self.assertFalse(thread.is_alive())

    def test_handle_work_error(self):
        """Test that a work unit error is handled correctly."""
        work_unit = WorkUnit(trial_id='trial_1', type=WorkUnitType.TRAIN_EPOCH)
        error_msg = "Something went wrong"

        self.orchestrator._handle_work_error(work_unit, error_msg)

        self.mock_datastore_instance.update_trial_status.assert_called_once_with('trial_1', TrialStatus.PRUNED)

if __name__ == '__main__':
    unittest.main()
