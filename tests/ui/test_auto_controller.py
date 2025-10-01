import unittest
from unittest.mock import MagicMock, patch

from sde.ui.auto_controller import AutoController

# Mocking the main window and other UI components
class MockMainWindow(MagicMock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.view_model = MagicMock()
        # Ensure the controller doesn't think an experiment is already running
        self.view_model.experiment.status.is_running.return_value = False
        self.setup_pane = MagicMock()
        self.main_window = MagicMock()

    def append_log_message(self, message):
        pass

class TestAutoController(unittest.TestCase):

    def setUp(self):
        # Patch QTimer to avoid PyQt dependency issues in a non-GUI environment
        self.qtimer_patcher = patch('sde.ui.base_automation_controller.QTimer')
        self.mock_qtimer = self.qtimer_patcher.start()
        self.addCleanup(self.qtimer_patcher.stop)

        # Patch the registry for the duration of the test
        self.registry_patcher = patch('sde.ui.auto_controller.registry')
        self.mock_registry = self.registry_patcher.start()
        self.addCleanup(self.registry_patcher.stop)

        # Configure the mock registry to return predictable values
        self.mock_registry.list_challenges.return_value = ['Challenge1', 'Challenge2']
        self.mock_registry.list_models.return_value = ['ModelA', 'ModelB']

        self.main_window = MockMainWindow()
        self.auto_controller = AutoController(self.main_window)

    def test_initialization(self):
        """Test that the controller initializes correctly."""
        self.assertIsNotNone(self.auto_controller)
        self.assertTrue(hasattr(self.auto_controller, '_all_possible_experiments'))
        self.assertGreater(len(self.auto_controller._all_possible_experiments), 0)
        # Check that a QTimer was instantiated
        self.mock_qtimer.assert_called_once_with(self.main_window)

    def test_select_next_experiment_is_novel(self):
        """Test that the controller selects a novel experiment."""
        # Get the first experiment
        challenge1, models1 = self.auto_controller._select_next_experiment()
        # Add it to the history
        self.auto_controller._experiment_history.add((challenge1, frozenset(models1)))

        # Get the second experiment
        challenge2, models2 = self.auto_controller._select_next_experiment()

        # They should not be the same
        self.assertNotEqual((challenge1, frozenset(models1)), (challenge2, frozenset(models2)))

    def test_history_reset(self):
        """Test that the experiment history resets after all combinations are used."""
        # Manually fill the history to simulate all experiments being run
        all_experiments = self.auto_controller._all_possible_experiments
        self.auto_controller._experiment_history = set(all_experiments)
        self.assertEqual(len(self.auto_controller._experiment_history), len(all_experiments))

        # The next call to select_next_experiment should find no novel experiments,
        # which triggers the history reset logic.
        self.auto_controller._select_next_experiment()

        # The history should now be empty, ready for the next cycle.
        self.assertEqual(len(self.auto_controller._experiment_history), 0)

    def test_log_message_called_on_start(self):
        """Test that log messages are sent to the main window on start."""
        with patch.object(self.main_window, 'append_log_message') as mock_log:
            # We mock _run_new_experiment to isolate the start method's logging
            with patch.object(self.auto_controller, '_run_new_experiment'):
                self.auto_controller.start()
                # Check for the "Auto-discovery mode started." message
                mock_log.assert_any_call({'level': 'INFO', 'message': '[Auto Mode] Auto-discovery mode started.'})

if __name__ == '__main__':
    unittest.main()