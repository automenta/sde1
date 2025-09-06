import unittest
import pytest
from unittest.mock import patch

# All Qt imports must be here
from PyQt6.QtWidgets import QApplication, QPushButton
from PyQt6.QtCore import Qt

# Now import the class to be tested
from sde.ui.main_window import MainWindow

@pytest.mark.skip(reason="UI tests require a running X server and cannot be run in a headless environment without xvfb.")
class TestV2UIMainWindow(unittest.TestCase):

    @patch('sde.ui.main_window.ExperimentOrchestrator')
    def setUp(self, MockOrchestrator):
        """Set up the test environment before each test."""
        # Create a QApplication instance for the test run.
        # This is moved here from the global scope to prevent crashes during
        # test collection in a headless environment.
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        self.app = app

        # Mock the orchestrator to isolate the UI
        self.mock_orchestrator = MockOrchestrator.return_value
        # Instantiate the main window
        self.main_win = MainWindow()

    def tearDown(self):
        """Clean up after each test."""
        # This can be important in more complex tests to avoid side effects
        pass

    def test_initial_ui_state(self):
        """Test that the UI starts in the correct initial, locked-down state."""
        # Initial state is DEFINING, no challenge set
        initial_state = {
            "status": "DEFINING",
            "challenge": None,
            "algorithms": {},
            "trials": {},
            "valid_actions": {
                "global": ["SET_CHALLENGE"],
                "algorithms": {},
                "trials": {}
            }
        }
        self.main_win.on_state_changed(initial_state)

        # Assertions
        self.assertTrue(self.main_win.dataset_combo.isEnabled(), "Dataset combo should be enabled to set a challenge.")
        self.assertFalse(self.main_win.model_list.isEnabled(), "Model list should be disabled before challenge is set.")
        self.assertFalse(self.main_win.start_button.isEnabled(), "Start button should be disabled.")
        self.assertFalse(self.main_win.tune_button.isEnabled(), "Tune button should be disabled.")
        self.assertFalse(self.main_win.pause_button.isEnabled(), "Pause button should be disabled.")
        self.assertEqual(self.main_win.algorithms_table.rowCount(), 0, "Algorithms table should be empty.")

    def test_ui_state_after_challenge_set(self):
        """Test UI state after a challenge has been selected."""
        state = {
            "status": "DEFINING",
            "challenge": {"name": "MNIST"},
            "algorithms": {},
            "trials": {},
            "valid_actions": {
                "global": ["ADD_ALGORITHM", "SET_ADAPTIVE_POLICY", "SET_BUDGET"],
                "algorithms": {}, "trials": {}
            }
        }
        self.main_win.on_state_changed(state)

        # Assertions
        self.assertFalse(self.main_win.dataset_combo.isEnabled(), "Dataset combo should be locked after setting a challenge.")
        self.assertTrue(self.main_win.model_list.isEnabled(), "Model list should be enabled to add an algorithm.")
        self.assertFalse(self.main_win.start_button.isEnabled(), "Start button should be disabled until an algorithm is added.")

    def test_ui_state_after_algorithm_added(self):
        """Test UI state after an algorithm has been added."""
        state = {
            "status": "DEFINING",
            "challenge": {"name": "MNIST"},
            "algorithms": {"algo_0": {"id": "algo_0", "name": "TestCNN"}},
            "trials": {},
            "valid_actions": {
                "global": ["ADD_ALGORITHM", "SET_ADAPTIVE_POLICY", "SET_BUDGET", "START_RUN"],
                "algorithms": {"algo_0": ["UPDATE_PARAM_SPACE", "REMOVE_ALGORITHM"]},
                "trials": {}
            }
        }
        self.main_win.on_state_changed(state)

        # Assertions
        self.assertTrue(self.main_win.start_button.isEnabled(), "Start button should be enabled now.")
        self.assertTrue(self.main_win.tune_button.isEnabled(), "Tune button should be enabled now.")
        self.assertEqual(self.main_win.algorithms_table.rowCount(), 1, "Algorithms table should have one entry.")
        self.assertEqual(self.main_win.algorithms_table.item(0, 0).text(), "TestCNN")

        # Check the remove button in the table
        remove_button = self.main_win.algorithms_table.cellWidget(0, 1).findChild(QPushButton)
        self.assertIsNotNone(remove_button)
        self.assertTrue(remove_button.isEnabled(), "Remove button for the algorithm should be enabled.")

    def test_ui_state_when_running(self):
        """Test UI state when the experiment is actively running."""
        state = {
            "status": "RUNNING",
            "challenge": {"name": "MNIST"},
            "algorithms": {"algo_0": {"id": "algo_0", "name": "TestCNN"}},
            "trials": {"trial_0": {
                "id": "trial_0",
                "status": "ACTIVE",
                "algorithm_name": "TestCNN",
                "current_epoch": 1,
                "results": {},
                "hyperparameters": {"lr": 0.01}
            }},
            "valid_actions": {
                "global": ["PAUSE_RUN", "ADD_ALGORITHM", "SET_BUDGET"],
                "algorithms": {"algo_0": ["UPDATE_PARAM_SPACE", "REMOVE_ALGORITHM"]},
                "trials": {"trial_0": ["MANUAL_PRUNE_TRIAL"]}
            }
        }
        self.main_win.on_state_changed(state)

        # Assertions
        self.assertFalse(self.main_win.start_button.isEnabled(), "Start button should be disabled when running.")
        self.assertTrue(self.main_win.pause_button.isEnabled(), "Pause button should be enabled when running.")
        self.assertFalse(self.main_win.resume_button.isEnabled(), "Resume button should be disabled.")
        self.assertTrue(self.main_win.add_models_button.isEnabled(), "Add models button should be enabled mid-run.")

    def test_remove_algorithm_dispatch(self):
        """Test that the 'Remove' button in the algorithm table dispatches the correct action."""
        # 1. Set up the state where removing is possible
        state = {
            "status": "DEFINING",
            "challenge": {"name": "MNIST"},
            "algorithms": {"algo_0": {"id": "algo_0", "name": "TestCNN"}},
            "trials": {},
            "valid_actions": {
                "global": ["START_RUN"],
                "algorithms": {"algo_0": ["REMOVE_ALGORITHM"]},
                "trials": {}
            }
        }
        self.main_win.on_state_changed(state)

        # 2. Find and click the button
        remove_button = self.main_win.algorithms_table.cellWidget(0, 1).findChild(QPushButton)
        self.assertIsNotNone(remove_button)
        remove_button.click()

        # 3. Assert that dispatch was called with the correct payload
        self.mock_orchestrator.dispatch.assert_called_once_with("REMOVE_ALGORITHM", {"algorithm_id": "algo_0"})

if __name__ == '__main__':
    unittest.main()
