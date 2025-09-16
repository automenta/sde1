import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

from sde.core.definitions import DatasetType
from sde.core.domain import ExecutionSettings
from sde.core.domain import Experiment
from sde.core.domain import ExperimentStatus
from sde.core.domain import Trial
from sde.core.domain import TrialStatus
from sde.core.domain import WorkUnit
from sde.core.domain import WorkUnitType
from sde.discovery import discover_and_register_components
from sde.engine.runtime import SdeRuntimeEngine
from sde.exploration.schedulers import SuccessiveHalvingScheduler


class TestResumeLogic(unittest.TestCase):

    @patch("sde.engine.runtime.ComputeScheduler")
    @patch("sde.engine.runtime.SchedulerFactory")
    def test_resume_from_saved_state(self, MockSchedulerFactory, MockComputeScheduler):
        # 1. Create an Experiment object that looks like it was saved mid-run
        discover_and_register_components()
        experiment = Experiment(id="test_exp_1")
        experiment.status = ExperimentStatus.PAUSED
        experiment.challenge = {
            "name": "CIFAR-10",
            "type": DatasetType.IMAGE_CLASSIFICATION.value,
        }
        experiment.adaptive_policy = "SuccessiveHalving"
        experiment.execution_settings = ExecutionSettings(
            num_workers=1,
            num_trials_per_algo=5,
            enable_checkpointing=False,
            work_unit_timeout_seconds=300,
        )

        # 2. Create some trials in various states
        trials = {
            "trial_1": Trial(
                id="trial_1",
                algorithm_name="ResNet",
                hyperparameters={},
                status=TrialStatus.COMPLETED,
                current_epoch=10,
            ),
            "trial_2": Trial(
                id="trial_2",
                algorithm_name="ResNet",
                hyperparameters={},
                status=TrialStatus.ACTIVE,
                current_epoch=5,
                results={"accuracy": [(1, 0.5), (5, 0.8)]},
            ),
            "trial_3": Trial(
                id="trial_3",
                algorithm_name="ResNet",
                hyperparameters={},
                status=TrialStatus.PRUNED,
                current_epoch=2,
            ),
            "trial_4": Trial(
                id="trial_4",
                algorithm_name="ResNet",
                hyperparameters={},
                status=TrialStatus.ACTIVE,
                current_epoch=5,
                results={"accuracy": [(1, 0.6), (5, 0.85)]},
            ),
            "trial_5": Trial(
                id="trial_5",
                algorithm_name="ResNet",
                hyperparameters={},
                status=TrialStatus.PENDING,
            ),
        }
        experiment.trials = trials

        # 3. Set up mocks for the engine's dependencies
        mock_adaptive_scheduler = MagicMock(spec=SuccessiveHalvingScheduler)
        mock_adaptive_scheduler.metric = "accuracy"
        mock_adaptive_scheduler.increasing = True
        mock_adaptive_scheduler.rehydrate_work_units.return_value = [
            WorkUnit(trial_id="trial_2", type=WorkUnitType.TRAIN_EPOCH, payload={}),
            WorkUnit(trial_id="trial_4", type=WorkUnitType.TRAIN_EPOCH, payload={}),
        ]
        MockSchedulerFactory.create_scheduler.return_value = mock_adaptive_scheduler

        # 4. Create the Runtime Engine
        runtime_engine = SdeRuntimeEngine(event_callback=lambda x, y: None)

        # 5. Initialize the engine by sending the START_RUN command
        payload = {
            "experiment_definition": experiment.to_dict(),
            "execution_settings": experiment.execution_settings.to_dict(),
            "start_paused": True,
        }
        runtime_engine._handle_start_run(payload)

        # 6. Check that the rehydration method was called on the scheduler
        mock_adaptive_scheduler.rehydrate_work_units.assert_called_once()

        # 7. Check the work queue
        work_queue = runtime_engine.work_queue
        self.assertEqual(work_queue.qsize(), 2)

        work_units = []
        while not work_queue.empty():
            work_units.append(work_queue.get()[2])  # [2] to get the WorkUnit object

        active_trial_ids = {"trial_2", "trial_4"}
        work_unit_trial_ids = {wu.trial_id for wu in work_units}
        self.assertEqual(active_trial_ids, work_unit_trial_ids)

        # Clean up the engine
        runtime_engine._handle_stop_run({})


if __name__ == "__main__":
    unittest.main()
