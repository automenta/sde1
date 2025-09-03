import unittest
from unittest.mock import MagicMock, patch
import time

from sde.engine.scheduler import Scheduler
from sde.core.types import Trial, TrialStatus, WorkUnit, WorkUnitType
from sde.exploration.schedulers import SuccessiveHalvingScheduler

class TestScheduler(unittest.TestCase):

    def setUp(self):
        """Set up a basic scheduler with mock components for testing."""
        self.trials = [
            Trial(id='trial_1', algorithm_name='Algo1', hyperparameters={}),
            Trial(id='trial_2', algorithm_name='Algo2', hyperparameters={}),
        ]
        self.adaptive_scheduler = MagicMock(spec=SuccessiveHalvingScheduler)
        self.adaptive_scheduler.metric = "accuracy"
        self.adaptive_scheduler.increasing = True
        self.adaptive_scheduler.get_initial_work_units.return_value = []
        self.adaptive_scheduler.get_next_work_units.return_value = []

        self.scheduler = Scheduler(
            trials=self.trials,
            dataset_name="TestDataset",
            adaptive_scheduler=self.adaptive_scheduler,
            max_workers=2,
            enable_checkpointing=False
        )
        # Manually mock the signal callbacks
        self.scheduler.log_message.connect(MagicMock())
        self.scheduler.trial_updated.connect(MagicMock())
        self.scheduler.experiment_finished.connect(MagicMock())
        self.scheduler.insight_generated.connect(MagicMock())


    def test_initialization(self):
        """Test that the scheduler initializes correctly."""
        self.assertEqual(len(self.scheduler.trials), 2)
        self.assertIsInstance(self.scheduler.trials['trial_1'], Trial)
        self.assertFalse(self.scheduler._is_paused)
        self.assertTrue(self.scheduler._is_running)

    def test_pause_and_resume(self):
        """Test the pause and resume functionality."""
        log_callback = self.scheduler.log_message._callbacks[0]

        self.scheduler.pause()
        self.assertTrue(self.scheduler._is_paused)
        log_callback.assert_called_with("INFO: Pausing experiment. Finishing active work...")

        self.scheduler.resume()
        self.assertFalse(self.scheduler._is_paused)
        log_callback.assert_called_with("INFO: Resuming experiment...")

    def test_throttle(self):
        """Test the worker throttle functionality."""
        log_callback = self.scheduler.log_message._callbacks[0]

        self.scheduler.set_throttle(50)
        self.assertEqual(self.scheduler._active_workers, 1)
        log_callback.assert_called_with("INFO: Throttle set to 50%. Active workers: 1/2")

        self.scheduler.set_throttle(100)
        self.assertEqual(self.scheduler._active_workers, 2)
        log_callback.assert_called_with("INFO: Throttle set to 100%. Active workers: 2/2")

        self.scheduler.set_throttle(150)
        log_callback.assert_called_with("WARN: Throttle percentage must be between 1-100. Got 150.")

    @patch('sde.engine.scheduler.concurrent.futures.wait')
    @patch('sde.engine.scheduler.concurrent.futures.ProcessPoolExecutor')
    def test_main_loop_dispatches_and_processes_work(self, mock_executor, mock_wait):
        """Test the main loop with a mock executor to ensure work is dispatched and processed."""
        # --- Setup Mocks ---
        mock_pool = mock_executor.return_value
        self.scheduler.executor = mock_pool # Manually set the executor
        mock_future = MagicMock()
        mock_future.result.return_value = {
            'state_updates': {'current_epoch': 1, 'checkpoint_path': None},
            'metrics': {'accuracy': 0.9}
        }
        mock_pool.submit.return_value = mock_future
        mock_wait.return_value = ([mock_future], []) # Simulate immediate completion

        # --- Setup Scheduler State ---
        work_unit = WorkUnit(trial_id='trial_1', type=WorkUnitType.TRAIN_EPOCH)
        self.scheduler.work_queue = [work_unit]
        self.scheduler.trials['trial_1'].status = TrialStatus.ACTIVE
        self.scheduler._is_running = True # Ensure the loop runs

        # --- Get Signal Callbacks ---
        log_callback = self.scheduler.log_message._callbacks[0]
        trial_updated_callback = self.scheduler.trial_updated._callbacks[0]

        # --- Run one iteration of the loop ---
        # We manually call the internal methods to simulate one pass of the main loop
        self.scheduler._dispatch_work(self.scheduler.max_workers)
        self.scheduler.running_futures = {mock_future: work_unit} # Manually add future
        self.scheduler._process_completed_work()

        # --- Assertions ---
        mock_pool.submit.assert_called_once()
        # Check that the correct arguments were passed to the worker process
        submit_args = mock_pool.submit.call_args[0]
        self.assertEqual(submit_args[1], work_unit) # work_unit
        self.assertEqual(submit_args[2], self.scheduler.trials['trial_1']) # trial

        log_callback.assert_any_call("INFO: Dispatched: TRAIN_EPOCH for Trial trial_1, Epoch 1")
        log_callback.assert_any_call("Result for Trial trial_1, Epoch 1: {'accuracy': 0.9}")
        trial_updated_callback.assert_any_call(self.scheduler.trials['trial_1'].to_dict())
        self.adaptive_scheduler.get_next_work_units.assert_called_once()

if __name__ == '__main__':
    unittest.main()
