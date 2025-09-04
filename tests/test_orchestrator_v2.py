import unittest
from unittest.mock import MagicMock, patch, ANY

from sde.engine.orchestrator import ExperimentOrchestrator
from sde.core.types import ExperimentStatus

# By patching Signal.emit at the class level, we replace the method with a
# single mock for the entire lifetime of the test class. This solves the
# problem of the signals being instantiated as class attributes before we
# can mock them in a setUp method.
@patch('sde.engine.orchestrator.Signal.emit')
class TestExperimentOrchestrator(unittest.TestCase):

    def test_initialization(self, mock_emit):
        """Test that the orchestrator starts in the DEFINING state."""
        orchestrator = ExperimentOrchestrator()

        self.assertEqual(orchestrator.experiment.status, ExperimentStatus.DEFINING)
        self.assertIsNone(orchestrator.experiment.challenge)
        self.assertEqual(orchestrator.experiment.algorithms, {})

        # __init__ should log its creation but not emit a state change.
        mock_emit.assert_called_once_with("INFO: Orchestrator initialized in DEFINING state.")

    def test_get_valid_actions(self, mock_emit):
        """Test valid actions in various states."""
        orchestrator = ExperimentOrchestrator()

        # Initially, only setting a challenge is valid
        self.assertEqual(orchestrator.get_valid_actions(), ["SET_CHALLENGE"])

        # After setting a challenge, adding an algorithm is valid
        orchestrator.experiment.challenge = {"name": "Test Challenge"}
        self.assertEqual(orchestrator.get_valid_actions(), ["ADD_ALGORITHM"])

        # After adding an algorithm, starting a run is also valid
        orchestrator.experiment.algorithms['algo1'] = MagicMock()
        self.assertEqual(orchestrator.get_valid_actions(), ["ADD_ALGORITHM", "START_RUN"])

        # In a running state
        orchestrator.experiment.status = ExperimentStatus.RUNNING
        self.assertEqual(orchestrator.get_valid_actions(), ["PAUSE_RUN", "ADD_ALGORITHM"])

    def test_dispatch_set_challenge(self, mock_emit):
        """Test the SET_CHALLENGE action."""
        orchestrator = ExperimentOrchestrator()
        mock_emit.reset_mock() # Reset after __init__ call

        challenge_payload = {"name": "CIFAR-10", "metric": "accuracy"}
        orchestrator.dispatch("SET_CHALLENGE", challenge_payload)

        self.assertEqual(orchestrator.experiment.challenge, challenge_payload)
        # It should log the action AND emit a state change
        self.assertEqual(mock_emit.call_count, 2)
        mock_emit.assert_any_call("INFO: Challenge set to 'CIFAR-10'")
        # The other call is the state_changed signal, which passes a dict
        mock_emit.assert_any_call(ANY)
        self.assertTrue(any(isinstance(call.args[0], dict) for call in mock_emit.call_args_list))

    def test_dispatch_start_run(self, mock_emit):
        """Test the START_RUN action."""
        orchestrator = ExperimentOrchestrator()
        orchestrator.experiment.challenge = {"name": "Test"}
        orchestrator.experiment.algorithms['algo1'] = MagicMock()
        mock_emit.reset_mock()

        orchestrator.dispatch("START_RUN", {})
        self.assertEqual(orchestrator.experiment.status, ExperimentStatus.RUNNING)
        # Check for the log messages
        mock_emit.assert_any_call("INFO: START_RUN action received. Initializing runtime.")
        mock_emit.assert_any_call("SIM: Would start SdeRuntimeEngine now.")
        # Check that a state change was emitted
        self.assertTrue(any(isinstance(call.args[0], dict) for call in mock_emit.call_args_list))

    def test_dispatch_invalid_action(self, mock_emit):
        """Test that invalid actions are logged and ignored."""
        orchestrator = ExperimentOrchestrator()
        mock_emit.reset_mock()

        # Try to add an algorithm before setting a challenge
        orchestrator.dispatch("ADD_ALGORITHM", {})

        self.assertEqual(len(orchestrator.experiment.algorithms), 0)
        mock_emit.assert_called_once_with("WARN: Action 'ADD_ALGORITHM' is not valid in state 'DEFINING'")

        mock_emit.reset_mock()

        # Start the run, then try to set the challenge again
        orchestrator.experiment.status = ExperimentStatus.RUNNING
        orchestrator.dispatch("SET_CHALLENGE", {"name": "Another"})
        mock_emit.assert_called_once_with("WARN: Action 'SET_CHALLENGE' is not valid in state 'RUNNING'")
