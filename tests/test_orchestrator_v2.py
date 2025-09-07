import unittest
from unittest.mock import MagicMock, patch, ANY

from sde.engine.orchestrator import ExperimentOrchestrator
from sde.engine.action_validator import ActionValidator
from sde.core.types import ExperimentStatus, Trial, TrialStatus, AlgorithmConfig
from sde.core.actions import ActionType

@patch('sde.engine.orchestrator.Signal.emit')
class TestExperimentOrchestrator(unittest.TestCase):

    def test_initialization(self, mock_emit):
        """Test that the orchestrator starts in the DEFINING state."""
        orchestrator = ExperimentOrchestrator()
        self.assertEqual(orchestrator.experiment.status, ExperimentStatus.DEFINING)
        self.assertIsNone(orchestrator.experiment.challenge)
        self.assertEqual(orchestrator.experiment.algorithms, {})
        mock_emit.assert_called_with("INFO: Orchestrator initialized in DEFINING state.")

    def test_get_valid_actions_structured(self, mock_emit):
        """Test the new structured output of get_valid_actions."""
        orchestrator = ExperimentOrchestrator()
        mock_emit.reset_mock()

        actions = ActionValidator.get_valid_actions(orchestrator.experiment)
        self.assertEqual(actions['global'], ["SET_CHALLENGE"])

        orchestrator.experiment.challenge = {"name": "Test Challenge"}
        actions = ActionValidator.get_valid_actions(orchestrator.experiment)
        self.assertIn("ADD_ALGORITHM", actions['global'])

        algo_config = AlgorithmConfig(id='algo1', name='TestAlgo', parameter_space={})
        orchestrator.experiment.algorithms['algo1'] = algo_config
        actions = ActionValidator.get_valid_actions(orchestrator.experiment)
        self.assertIn("START_RUN", actions['global'])
        self.assertIn("UPDATE_PARAM_SPACE", actions['algorithms']['algo1'])

        orchestrator.experiment.status = ExperimentStatus.RUNNING
        trial = Trial(id='trial1', algorithm_name='TestAlgo', hyperparameters={}, status=TrialStatus.ACTIVE, results=[(1, 0.5)])
        orchestrator.experiment.trials['trial1'] = trial
        actions = ActionValidator.get_valid_actions(orchestrator.experiment)
        self.assertIn("PAUSE_RUN", actions['global'])
        self.assertIn("MANUAL_PRUNE_TRIAL", actions['trials']['trial1'])

    def test_dispatch_set_adaptive_policy(self, mock_emit):
        """Test the SET_ADAPTIVE_POLICY action."""
        orchestrator = ExperimentOrchestrator()
        mock_emit.reset_mock()
        orchestrator.experiment.challenge = {"name": "Test"}
        orchestrator.dispatch(ActionType.SET_ADAPTIVE_POLICY, {"policy_name": "Hyperband"})
        self.assertEqual(orchestrator.experiment.adaptive_policy, "Hyperband")
        mock_emit.assert_any_call("INFO: Adaptive policy set to 'Hyperband'.")

    def test_dispatch_remove_algorithm(self, mock_emit):
        """Test removing an algorithm and pruning its trials."""
        orchestrator = ExperimentOrchestrator()
        mock_emit.reset_mock()
        algo = AlgorithmConfig(id='algo1', name='TestAlgo', parameter_space={})
        orchestrator.experiment.algorithms['algo1'] = algo
        t1 = Trial(id='t1', algorithm_name='TestAlgo', hyperparameters={})
        orchestrator.experiment.trials = {'t1': t1}
        orchestrator.dispatch(ActionType.REMOVE_ALGORITHM, {"algorithm_id": "algo1"})
        self.assertNotIn('algo1', orchestrator.experiment.algorithms)
        self.assertEqual(orchestrator.experiment.trials['t1'].status, TrialStatus.PRUNED)

    def test_dispatch_manual_prune_trial(self, mock_emit):
        """Test manually pruning a single trial."""
        orchestrator = ExperimentOrchestrator()
        mock_emit.reset_mock()
        t1 = Trial(id='t1', algorithm_name='TestAlgo', hyperparameters={}, status=TrialStatus.ACTIVE)
        orchestrator.experiment.trials = {'t1': t1}
        orchestrator.experiment.status = ExperimentStatus.RUNNING
        orchestrator.dispatch(ActionType.MANUAL_PRUNE_TRIAL, {"trial_id": "t1"})
        self.assertEqual(orchestrator.experiment.trials['t1'].status, TrialStatus.PRUNED)

    def test_dispatch_spawn_similar_trial(self, mock_emit):
        """Test spawning a new trial from an existing one."""
        orchestrator = ExperimentOrchestrator()
        mock_emit.reset_mock()
        source_trial = Trial(id='t1', algorithm_name='TestAlgo', hyperparameters={'lr': 0.1}, results=[(1, 0.9)])
        orchestrator.experiment.trials = {'t1': source_trial}
        orchestrator.experiment.status = ExperimentStatus.RUNNING
        orchestrator.dispatch(ActionType.SPAWN_SIMILAR_TRIAL, {"source_trial_id": "t1"})
        self.assertEqual(len(orchestrator.experiment.trials), 2)
        new_trial = next(t for t in orchestrator.experiment.trials.values() if t.id != 't1')
        self.assertEqual(new_trial.hyperparameters['lr'], 0.1)

    @patch('sde.engine.orchestrator.SdeRuntimeEngine')
    def test_runtime_calls_on_prune(self, MockSdeRuntimeEngine, mock_emit):
        """Test that pruning a trial calls the runtime engine to cancel work."""
        orchestrator = ExperimentOrchestrator()
        mock_emit.reset_mock()
        orchestrator.experiment.status = ExperimentStatus.RUNNING
        orchestrator.runtime_engine = MockSdeRuntimeEngine()
        t1 = Trial(id='t1', algorithm_name='TestAlgo', hyperparameters={}, status=TrialStatus.ACTIVE)
        orchestrator.experiment.trials = {'t1': t1}
        orchestrator.dispatch(ActionType.MANUAL_PRUNE_TRIAL, {"trial_id": "t1"})
        orchestrator.runtime_engine.cancel_work_for_trial.assert_called_once_with("t1")

    @patch('sde.engine.orchestrator.SdeRuntimeEngine')
    def test_runtime_calls_on_spawn(self, MockSdeRuntimeEngine, mock_emit):
        """Test that spawning a trial calls the runtime engine to add it live."""
        orchestrator = ExperimentOrchestrator()
        mock_emit.reset_mock()
        orchestrator.experiment.status = ExperimentStatus.RUNNING
        orchestrator.runtime_engine = MockSdeRuntimeEngine()
        source_trial = Trial(id='t1', algorithm_name='TestAlgo', hyperparameters={'lr': 0.1}, results=[(1, 0.9)])
        orchestrator.experiment.trials = {'t1': source_trial}
        orchestrator.dispatch(ActionType.SPAWN_SIMILAR_TRIAL, {"source_trial_id": "t1"})
        new_trial = next(t for t in orchestrator.experiment.trials.values() if t.id != 't1')
        orchestrator.runtime_engine.add_trials_live.assert_called_once_with([new_trial])

    @patch('sde.engine.orchestrator.SdeRuntimeEngine')
    def test_dispatch_start_run(self, MockSdeRuntimeEngine, mock_emit):
        """Test the START_RUN action."""
        orchestrator = ExperimentOrchestrator()
        mock_emit.reset_mock()
        orchestrator.experiment.challenge = {"name": "MNIST", "performance_metric_name": "accuracy"}
        algo = AlgorithmConfig(id='algo1', name='TestAlgo', parameter_space={'lr': (0.01, 0.1)})
        orchestrator.experiment.algorithms['algo1'] = algo
        start_payload = {"num_workers": 2, "num_trials_per_algo": 5}
        orchestrator.dispatch(ActionType.START_RUN, start_payload)
        self.assertEqual(orchestrator.experiment.status, ExperimentStatus.RUNNING)
        MockSdeRuntimeEngine.assert_called_once()
        mock_engine_instance = MockSdeRuntimeEngine.return_value
        mock_engine_instance.start.assert_called_once()

    @patch('sde.engine.orchestrator.SdeRuntimeEngine')
    def test_add_algorithm_mid_run(self, MockSdeRuntimeEngine, mock_emit):
        """Test adding a new algorithm to a running experiment."""
        orchestrator = ExperimentOrchestrator()
        orchestrator.experiment.challenge = {"name": "MNIST", "performance_metric_name": "accuracy"}
        algo1 = AlgorithmConfig(id='algo1', name='TestAlgo1', parameter_space={'lr': (0.01, 0.1)})
        orchestrator.experiment.algorithms['algo1'] = algo1
        start_payload = {"num_workers": 2, "num_trials_per_algo": 5}
        orchestrator.dispatch(ActionType.START_RUN, start_payload)
        self.assertEqual(orchestrator.experiment.status, ExperimentStatus.RUNNING)
        mock_engine_instance = MockSdeRuntimeEngine.return_value
        orchestrator.runtime_engine = mock_engine_instance

        mock_scheduler = MagicMock()
        new_mock_trial = Trial(id='trial_new_1', algorithm_name='TestAlgo2', hyperparameters={'lr': 0.5})
        mock_scheduler.generate_initial_trials.return_value = [new_mock_trial]
        mock_engine_instance.adaptive_scheduler = mock_scheduler

        trials_before = len(orchestrator.experiment.trials)
        self.assertGreater(trials_before, 0)

        mock_emit.reset_mock()
        add_payload = {"name": "TestAlgo2", "parameter_space": {"lr": (0.2, 0.9)}}
        orchestrator.dispatch(ActionType.ADD_ALGORITHM, add_payload)

        self.assertIn('algo_1', orchestrator.experiment.algorithms)
        self.assertEqual(orchestrator.experiment.algorithms['algo_1'].name, "TestAlgo2")

        trials_after = len(orchestrator.experiment.trials)
        self.assertGreater(trials_after, trials_before)

        newly_added_trials = [t for t in orchestrator.experiment.trials.values() if t.algorithm_name == "TestAlgo2"]
        self.assertGreater(len(newly_added_trials), 0)

        mock_engine_instance.add_trials_live.assert_called_once()
        self.assertEqual(mock_engine_instance.add_trials_live.call_args[0][0], newly_added_trials)

    @patch('sde.engine.orchestrator.SdeRuntimeEngine')
    def test_pause_and_resume_run(self, MockSdeRuntimeEngine, mock_emit):
        """Test pausing and resuming a run with the new event-based mechanism."""
        orchestrator = ExperimentOrchestrator()
        orchestrator.experiment.challenge = {"name": "MNIST"}
        orchestrator.experiment.algorithms['algo1'] = AlgorithmConfig(id='a1', name='A1', parameter_space={})

        start_payload = {"num_workers": 2, "num_trials_per_algo": 5}
        orchestrator.dispatch(ActionType.START_RUN, start_payload)
        self.assertEqual(orchestrator.experiment.status, ExperimentStatus.RUNNING)
        mock_engine_instance = MockSdeRuntimeEngine.return_value
        orchestrator.runtime_engine = mock_engine_instance
        mock_engine_instance.start.assert_called_once()

        orchestrator.dispatch(ActionType.PAUSE_RUN, {})
        self.assertEqual(orchestrator.experiment.status, ExperimentStatus.PAUSED)
        mock_engine_instance.pause.assert_called_once()

        orchestrator.dispatch(ActionType.RESUME_RUN, {})
        self.assertEqual(orchestrator.experiment.status, ExperimentStatus.RUNNING)
        mock_engine_instance.resume.assert_called_once()

        mock_engine_instance.start.assert_called_once()

    @patch('sde.engine.orchestrator.SdeRuntimeEngine')
    def test_set_adaptive_policy_mid_run(self, MockSdeRuntimeEngine, mock_emit):
        """Test that changing the policy mid-run calls the runtime engine."""
        orchestrator = ExperimentOrchestrator()
        orchestrator.experiment.challenge = {"name": "MNIST", "performance_metric_name": "accuracy"}
        orchestrator.experiment.algorithms['algo1'] = AlgorithmConfig(id='a1', name='A1', parameter_space={})
        start_payload = {"num_workers": 2, "num_trials_per_algo": 5}
        orchestrator.dispatch(ActionType.START_RUN, start_payload)
        mock_engine_instance = MockSdeRuntimeEngine.return_value
        orchestrator.runtime_engine = mock_engine_instance
        self.assertEqual(orchestrator.experiment.adaptive_policy, "SuccessiveHalving")

        mock_emit.reset_mock()
        orchestrator.dispatch(ActionType.SET_ADAPTIVE_POLICY, {"policy_name": "Hyperband"})

        self.assertEqual(orchestrator.experiment.adaptive_policy, "Hyperband")
        mock_emit.assert_any_call("INFO: Adaptive policy set to 'Hyperband'.")

        mock_engine_instance.update_adaptive_policy.assert_called_once()
        new_scheduler = mock_engine_instance.update_adaptive_policy.call_args[0][0]
        from sde.exploration.schedulers import HyperbandScheduler
        self.assertIsInstance(new_scheduler, HyperbandScheduler)

    @patch('sde.engine.orchestrator.SdeRuntimeEngine')
    def test_update_param_space_mid_run(self, MockSdeRuntimeEngine, mock_emit):
        """Test updating parameter space for a running experiment generates new trials."""
        orchestrator = ExperimentOrchestrator()
        orchestrator.experiment.challenge = {"name": "MNIST", "performance_metric_name": "accuracy"}
        algo1 = AlgorithmConfig(id='algo1', name='TestAlgo1', parameter_space={'lr': (0.01, 0.1)})
        orchestrator.experiment.algorithms['algo1'] = algo1
        start_payload = {"num_workers": 2, "num_trials_per_algo": 5}
        orchestrator.dispatch(ActionType.START_RUN, start_payload)
        mock_engine_instance = MockSdeRuntimeEngine.return_value
        orchestrator.runtime_engine = mock_engine_instance
        mock_scheduler = MagicMock()
        mock_engine_instance.adaptive_scheduler = mock_scheduler

        trials_before = len(orchestrator.experiment.trials)
        mock_emit.reset_mock()

        new_space = {'lr': (0.1, 0.5), 'epochs': [10, 20]}
        orchestrator.dispatch(ActionType.UPDATE_PARAM_SPACE, {"algorithm_id": "algo1", "new_space": new_space})

        self.assertEqual(orchestrator.experiment.algorithms['algo1'].parameter_space, new_space)
        mock_scheduler.generate_initial_trials.assert_called_once()
        self.assertEqual(mock_scheduler.generate_initial_trials.call_args[0][0][0].parameter_space, new_space)
        mock_engine_instance.add_trials_live.assert_called_once()

    def test_dispatch_invalid_action(self, mock_emit):
        """Test that invalid actions are logged and ignored."""
        orchestrator = ExperimentOrchestrator()
        mock_emit.reset_mock()
        orchestrator.dispatch(ActionType.ADD_ALGORITHM, {})
        self.assertEqual(len(orchestrator.experiment.algorithms), 0)
        mock_emit.assert_any_call("WARN: Action 'ADD_ALGORITHM' is not valid for the current state or payload.")

    def test_dispatch_action_with_invalid_context(self, mock_emit):
        """
        Test that an action that exists but is invalid for the given context
        is correctly rejected by the strict validator.
        """
        orchestrator = ExperimentOrchestrator()
        orchestrator.experiment.status = ExperimentStatus.RUNNING
        algo = AlgorithmConfig(id='algo1', name='TestAlgo', parameter_space={})
        orchestrator.experiment.algorithms['algo1'] = algo
        trial = Trial(id='trial1', algorithm_name='TestAlgo', hyperparameters={}, status=TrialStatus.ACTIVE)
        orchestrator.experiment.trials['trial1'] = trial
        mock_emit.reset_mock()

        orchestrator.dispatch(ActionType.MANUAL_PRUNE_TRIAL, {"algorithm_id": "algo1"})

        self.assertEqual(orchestrator.experiment.trials['trial1'].status, TrialStatus.ACTIVE)
        mock_emit.assert_any_call("WARN: Action 'MANUAL_PRUNE_TRIAL' is not valid for the current state or payload.")

    def test_dispatch_fictitious_action(self, mock_emit):
        """Test that a completely non-existent action is rejected."""
        orchestrator = ExperimentOrchestrator()
        mock_emit.reset_mock()

        with self.assertRaises(AttributeError):
            orchestrator.dispatch(ActionType.DO_A_BARREL_ROLL, {})
