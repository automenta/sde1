import unittest
from unittest.mock import MagicMock, patch, ANY

from sde.engine.orchestrator import ExperimentOrchestrator
from sde.core.types import ExperimentStatus, Trial, TrialStatus, AlgorithmConfig

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

        actions = orchestrator.get_valid_actions()
        self.assertEqual(actions['global'], ["SET_CHALLENGE"])

        orchestrator.experiment.challenge = {"name": "Test Challenge"}
        actions = orchestrator.get_valid_actions()
        self.assertIn("ADD_ALGORITHM", actions['global'])

        algo_config = AlgorithmConfig(id='algo1', name='TestAlgo', parameter_space={})
        orchestrator.experiment.algorithms['algo1'] = algo_config
        actions = orchestrator.get_valid_actions()
        self.assertIn("START_RUN", actions['global'])
        self.assertIn("UPDATE_PARAM_SPACE", actions['algorithms']['algo1'])

        orchestrator.experiment.status = ExperimentStatus.RUNNING
        trial = Trial(id='trial1', algorithm_name='TestAlgo', hyperparameters={}, status=TrialStatus.ACTIVE, results=[(1, 0.5)])
        orchestrator.experiment.trials['trial1'] = trial
        actions = orchestrator.get_valid_actions()
        self.assertIn("PAUSE_RUN", actions['global'])
        self.assertIn("MANUAL_PRUNE_TRIAL", actions['trials']['trial1'])

    def test_dispatch_set_adaptive_policy(self, mock_emit):
        """Test the SET_ADAPTIVE_POLICY action."""
        orchestrator = ExperimentOrchestrator()
        mock_emit.reset_mock()
        orchestrator.experiment.challenge = {"name": "Test"}
        orchestrator.dispatch("SET_ADAPTIVE_POLICY", {"policy_name": "Hyperband"})
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
        orchestrator.dispatch("REMOVE_ALGORITHM", {"algorithm_id": "algo1"})
        self.assertNotIn('algo1', orchestrator.experiment.algorithms)
        self.assertEqual(orchestrator.experiment.trials['t1'].status, TrialStatus.PRUNED)

    def test_dispatch_manual_prune_trial(self, mock_emit):
        """Test manually pruning a single trial."""
        orchestrator = ExperimentOrchestrator()
        mock_emit.reset_mock()
        t1 = Trial(id='t1', algorithm_name='TestAlgo', hyperparameters={}, status=TrialStatus.ACTIVE)
        orchestrator.experiment.trials = {'t1': t1}
        orchestrator.experiment.status = ExperimentStatus.RUNNING
        orchestrator.dispatch("MANUAL_PRUNE_TRIAL", {"trial_id": "t1"})
        self.assertEqual(orchestrator.experiment.trials['t1'].status, TrialStatus.PRUNED)

    def test_dispatch_spawn_similar_trial(self, mock_emit):
        """Test spawning a new trial from an existing one."""
        orchestrator = ExperimentOrchestrator()
        mock_emit.reset_mock()
        source_trial = Trial(id='t1', algorithm_name='TestAlgo', hyperparameters={'lr': 0.1}, results=[(1, 0.9)])
        orchestrator.experiment.trials = {'t1': source_trial}
        orchestrator.experiment.status = ExperimentStatus.RUNNING
        orchestrator.dispatch("SPAWN_SIMILAR_TRIAL", {"source_trial_id": "t1"})
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
        orchestrator.dispatch("MANUAL_PRUNE_TRIAL", {"trial_id": "t1"})
        orchestrator.runtime_engine.cancel_work_for_trial.assert_called_once_with("t1")

    @patch('sde.engine.orchestrator.SdeRuntimeEngine')
    def test_runtime_calls_on_spawn(self, MockSdeRuntimeEngine, mock_emit):
        """Test that spawning a trial calls the runtime engine to inject it."""
        orchestrator = ExperimentOrchestrator()
        mock_emit.reset_mock()
        orchestrator.experiment.status = ExperimentStatus.RUNNING
        orchestrator.runtime_engine = MockSdeRuntimeEngine()
        source_trial = Trial(id='t1', algorithm_name='TestAlgo', hyperparameters={'lr': 0.1}, results=[(1, 0.9)])
        orchestrator.experiment.trials = {'t1': source_trial}
        orchestrator.dispatch("SPAWN_SIMILAR_TRIAL", {"source_trial_id": "t1"})
        new_trial = next(t for t in orchestrator.experiment.trials.values() if t.id != 't1')
        orchestrator.runtime_engine.inject_trial.assert_called_once_with(new_trial)

    @patch('sde.engine.orchestrator.SdeRuntimeEngine')
    def test_dispatch_start_run(self, MockSdeRuntimeEngine, mock_emit):
        """Test the START_RUN action."""
        orchestrator = ExperimentOrchestrator()
        mock_emit.reset_mock()
        orchestrator.experiment.challenge = {"name": "MNIST", "performance_metric_name": "accuracy"}
        algo = AlgorithmConfig(id='algo1', name='TestAlgo', parameter_space={'lr': (0.01, 0.1)})
        orchestrator.experiment.algorithms['algo1'] = algo
        orchestrator.dispatch("START_RUN", {})
        self.assertEqual(orchestrator.experiment.status, ExperimentStatus.RUNNING)
        MockSdeRuntimeEngine.assert_called_once()
        mock_engine_instance = MockSdeRuntimeEngine.return_value
        mock_engine_instance.start.assert_called_once_with()

    @patch('sde.engine.orchestrator.SdeRuntimeEngine')
    def test_pause_and_resume_run(self, MockSdeRuntimeEngine, mock_emit):
        """Test the PAUSE_RUN and RESUME_RUN actions and state transitions."""
        orchestrator = ExperimentOrchestrator()
        mock_emit.reset_mock()

        # Setup for a runnable state
        orchestrator.experiment.challenge = {"name": "MNIST", "performance_metric_name": "accuracy"}
        algo = AlgorithmConfig(id='algo1', name='TestAlgo', parameter_space={'lr': (0.01, 0.1)})
        orchestrator.experiment.algorithms['algo1'] = algo
        orchestrator.dispatch("START_RUN", {})

        mock_engine_instance = MockSdeRuntimeEngine.return_value
        mock_engine_instance.start.assert_called_once_with()
        self.assertEqual(orchestrator.experiment.status, ExperimentStatus.RUNNING)

        # Test PAUSE
        orchestrator.dispatch("PAUSE_RUN", {})
        self.assertEqual(orchestrator.experiment.status, ExperimentStatus.PAUSED)
        mock_engine_instance.stop.assert_called_once()
        mock_emit.assert_any_call("INFO: Experiment paused.")

        # Test RESUME
        orchestrator.dispatch("RESUME_RUN", {})
        self.assertEqual(orchestrator.experiment.status, ExperimentStatus.RUNNING)
        mock_engine_instance.start.assert_called_with(is_resuming=True)
        mock_emit.assert_any_call("INFO: Experiment resumed.")

    @patch('sde.engine.orchestrator.SdeRuntimeEngine')
    def test_run_completes_via_callback(self, MockSdeRuntimeEngine, mock_emit):
        """Test that the run_completed_callback correctly updates state."""
        orchestrator = ExperimentOrchestrator()
        mock_emit.reset_mock()

        # Setup for a runnable state
        orchestrator.experiment.challenge = {"name": "MNIST", "performance_metric_name": "accuracy"}
        orchestrator.experiment.algorithms['algo1'] = AlgorithmConfig(id='a1', name='A', parameter_space={})
        orchestrator.dispatch("START_RUN", {})

        # Capture the callback passed to the engine
        mock_engine_instance = MockSdeRuntimeEngine.return_value
        init_kwargs = MockSdeRuntimeEngine.call_args.kwargs
        completion_callback = init_kwargs.get("run_completed_callback")

        self.assertIsNotNone(completion_callback)
        self.assertEqual(orchestrator.experiment.status, ExperimentStatus.RUNNING)

        # Simulate the engine finishing
        completion_callback()

        self.assertEqual(orchestrator.experiment.status, ExperimentStatus.COMPLETED)
        mock_emit.assert_any_call("INFO: Run completed.")

    def test_insights_generate_suggested_actions(self, mock_emit):
        """Test that insights are converted into suggested actions."""
        orchestrator = ExperimentOrchestrator()
        mock_emit.reset_mock()

        # Test plateau insight -> prune suggestion
        plateau_insight = {"type": "PLATEAU", "trial_ids": ["t1"], "message": "..."}
        orchestrator.on_insights_generated([plateau_insight])

        self.assertEqual(len(orchestrator.experiment.suggested_actions), 1)
        suggestion = orchestrator.experiment.suggested_actions[0]
        self.assertEqual(suggestion['action_type'], "MANUAL_PRUNE_TRIAL")
        self.assertEqual(suggestion['payload']['trial_id'], "t1")
        mock_emit.assert_any_call(f"SUGGESTION: {suggestion['message']}")

        # Test best performer -> prioritize suggestion
        orchestrator.experiment.suggested_actions = [] # Reset
        best_perf_insight = {"type": "BEST_PERFORMER", "trial_ids": ["t2"], "message": "..."}
        orchestrator.on_insights_generated([best_perf_insight])

        self.assertEqual(len(orchestrator.experiment.suggested_actions), 1)
        suggestion = orchestrator.experiment.suggested_actions[0]
        self.assertEqual(suggestion['action_type'], "MANUAL_PRIORITIZE_TRIAL")
        self.assertEqual(suggestion['payload']['trial_id'], "t2")

    def test_dispatch_invalid_action(self, mock_emit):
        """Test that invalid actions are logged and ignored."""
        orchestrator = ExperimentOrchestrator()
        mock_emit.reset_mock()
        orchestrator.dispatch("ADD_ALGORITHM", {})
        self.assertEqual(len(orchestrator.experiment.algorithms), 0)
        mock_emit.assert_any_call("WARN: Action 'ADD_ALGORITHM' is not valid for the current state or payload.")
