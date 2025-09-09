import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

from sde.core.actions import ActionType
from sde.core.types import AlgorithmConfig
from sde.core.types import ExecutionSettings
from sde.core.types import Experiment
from sde.core.types import ExperimentStatus
from sde.core.types import Trial
from sde.core.types import TrialStatus
from sde.engine.action_validator import ActionValidator
from sde.engine.orchestrator import ExperimentOrchestrator


@patch('sde.engine.orchestrator.Signal.emit')
@patch('sde.engine.orchestrator.EngineProxy')
class TestExperimentOrchestrator(unittest.TestCase):

    def test_initialization(self, MockProxy, mock_emit):
        """Test that the orchestrator initializes and creates a proxy."""
        orchestrator = ExperimentOrchestrator()
        self.assertEqual(orchestrator.experiment.status, ExperimentStatus.DEFINING)
        self.assertIsNotNone(orchestrator.engine_proxy)
        MockProxy.assert_called_once()
        mock_emit.assert_any_call({'level': 'INFO', 'message': 'Orchestrator initialized.'})

    def test_dispatch_start_run(self, MockProxy, mock_emit):
        """Test that START_RUN dispatches a command to the proxy."""
        orchestrator = ExperimentOrchestrator()
        orchestrator.experiment.challenge = {"name": "MNIST"}
        algo = AlgorithmConfig(id='algo1', name='TestAlgo', parameter_space={'lr': (0.01, 0.1)})
        orchestrator.experiment.algorithms['algo1'] = algo
        start_payload = {"num_workers": 2, "num_trials_per_algo": 5, "enable_checkpointing": False, "work_unit_timeout_seconds": 300}

        orchestrator.dispatch(ActionType.START_RUN, start_payload)

        self.assertEqual(orchestrator.experiment.status, ExperimentStatus.RUNNING)
        orchestrator.engine_proxy.post_command.assert_called_once()
        command_name = orchestrator.engine_proxy.post_command.call_args[0][0]
        command_payload = orchestrator.engine_proxy.post_command.call_args[0][1]
        self.assertEqual(command_name, "START_RUN")
        self.assertIn("experiment_definition", command_payload)

    def test_dispatch_pause_and_resume(self, MockProxy, mock_emit):
        """Test that PAUSE_RUN and RESUME_RUN dispatch commands."""
        orchestrator = ExperimentOrchestrator()
        orchestrator.experiment.status = ExperimentStatus.RUNNING

        orchestrator.dispatch(ActionType.PAUSE_RUN, {})
        self.assertEqual(orchestrator.experiment.status, ExperimentStatus.PAUSED)
        orchestrator.engine_proxy.post_command.assert_called_with("PAUSE_RUN")

        orchestrator.dispatch(ActionType.RESUME_RUN, {})
        self.assertEqual(orchestrator.experiment.status, ExperimentStatus.RUNNING)
        orchestrator.engine_proxy.post_command.assert_called_with("RESUME_RUN")

    def test_engine_event_updates_state(self, MockProxy, mock_emit):
        """Test that the orchestrator correctly processes an event from the engine."""
        orchestrator = ExperimentOrchestrator()
        mock_emit.reset_mock()

        # Simulate a TRIAL_UPDATED event from the engine
        updated_trial = Trial(id='t1', algorithm_name='TestAlgo', hyperparameters={}, status=TrialStatus.ACTIVE, current_epoch=1)
        event_payload = {"trial": updated_trial.to_dict()}
        orchestrator.on_engine_event("TRIAL_UPDATED", event_payload)

        # Check that the orchestrator's state was updated
        self.assertIn('t1', orchestrator.experiment.trials)
        self.assertEqual(orchestrator.experiment.trials['t1'].status, TrialStatus.ACTIVE)
        self.assertEqual(orchestrator.experiment.trials['t1'].current_epoch, 1)

        # Check that the UI was notified
        expected_state = orchestrator.experiment.to_dict()
        expected_state['valid_actions'] = ActionValidator.get_valid_actions(orchestrator.experiment)
        mock_emit.assert_any_call(expected_state)

    def test_dispatch_remove_algorithm_sends_command(self, MockProxy, mock_emit):
        """Test removing an algorithm sends the correct command."""
        orchestrator = ExperimentOrchestrator()
        algo = AlgorithmConfig(id='algo1', name='TestAlgo', parameter_space={})
        orchestrator.experiment.algorithms['algo1'] = algo
        orchestrator.dispatch(ActionType.REMOVE_ALGORITHM, {"algorithm_id": "algo1"})

        self.assertNotIn('algo1', orchestrator.experiment.algorithms)
        orchestrator.engine_proxy.post_command.assert_called_once_with("REMOVE_ALGORITHM", {"algorithm_id": "algo1"})

    def test_prune_trial_sends_command_and_updates_optimistically(self, MockProxy, mock_emit):
        """Test that pruning a trial sends a command and updates the UI optimistically."""
        orchestrator = ExperimentOrchestrator()
        t1 = Trial(id='t1', algorithm_name='TestAlgo', hyperparameters={}, status=TrialStatus.ACTIVE)
        orchestrator.experiment.trials = {'t1': t1}
        orchestrator.experiment.status = ExperimentStatus.RUNNING

        orchestrator.dispatch(ActionType.MANUAL_PRUNE_TRIAL, {"trial_id": "t1"})

        # The orchestrator should optimistically update its state for the UI
        self.assertEqual(orchestrator.experiment.trials['t1'].status, TrialStatus.PRUNED)
        # And it should send a command to the engine to do the real work
        orchestrator.engine_proxy.post_command.assert_called_once_with("PRUNE_TRIAL", {"trial_id": "t1"})

    def test_prioritize_trial_sends_command(self, MockProxy, mock_emit):
        """Test that prioritizing a trial sends a command."""
        orchestrator = ExperimentOrchestrator()
        t1 = Trial(id='t1', algorithm_name='TestAlgo', hyperparameters={}, status=TrialStatus.ACTIVE, priority=0)
        orchestrator.experiment.trials = {'t1': t1}

        orchestrator.dispatch(ActionType.MANUAL_PRIORITIZE_TRIAL, {"trial_id": "t1"})
        orchestrator.engine_proxy.post_command.assert_called_once_with("PRIORITIZE_TRIAL", {"trial_id": "t1"})

        # Simulate the confirmation event from the engine
        t1_updated = Trial(id='t1', algorithm_name='TestAlgo', hyperparameters={}, status=TrialStatus.ACTIVE, priority=10)
        orchestrator.on_engine_event("TRIAL_UPDATED", {"trial": t1_updated.to_dict()})
        self.assertEqual(orchestrator.experiment.trials['t1'].priority, 10)

    def test_spawn_trial_sends_command(self, MockProxy, mock_emit):
        """Test that spawning a trial sends a command."""
        orchestrator = ExperimentOrchestrator()
        source_trial = Trial(id='t1', algorithm_name='TestAlgo', hyperparameters={'lr': 0.1}, results={"accuracy": [(1, 0.9)]})
        orchestrator.experiment.trials = {'t1': source_trial}
        orchestrator.experiment.status = ExperimentStatus.RUNNING
        payload = {"source_trial_id": "t1", "new_hparams": {"lr": 0.2}}

        orchestrator.dispatch(ActionType.SPAWN_SIMILAR_TRIAL, payload)
        orchestrator.engine_proxy.post_command.assert_called_once_with("SPAWN_TRIAL", payload)

        # Simulate the new trial being created by the engine
        new_trial = Trial(id='t2', algorithm_name='TestAlgo', hyperparameters={'lr': 0.2})
        orchestrator.on_engine_event("TRIAL_UPDATED", {"trial": new_trial.to_dict()})
        self.assertEqual(len(orchestrator.experiment.trials), 2)
        self.assertIn('t2', orchestrator.experiment.trials)
