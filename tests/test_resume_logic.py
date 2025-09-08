import unittest
from sde.core.types import Experiment, Trial, TrialStatus, AlgorithmConfig, WorkUnitType
from sde.exploration.schedulers import SuccessiveHalvingScheduler
from sde.engine.runtime import SdeRuntimeEngine

class TestResumeLogic(unittest.TestCase):
    def test_resume_from_saved_state(self):
        # 1. Create an Experiment object that looks like it was saved mid-run
        experiment = Experiment(id="test_exp_1")
        experiment.status = "PAUSED"
        experiment.challenge = {"name": "CIFAR10", "type": "vision"}
        experiment.adaptive_policy = "SuccessiveHalving"
        experiment.execution_settings = {"num_workers": 1}

        # 2. Create some trials in various states
        trials = {
            "trial_1": Trial(id="trial_1", algorithm_name="ResNet", hyperparameters={}, status=TrialStatus.COMPLETED, current_epoch=10),
            "trial_2": Trial(id="trial_2", algorithm_name="ResNet", hyperparameters={}, status=TrialStatus.ACTIVE, current_epoch=5, results={"accuracy": [(1, 0.5), (5, 0.8)]}),
            "trial_3": Trial(id="trial_3", algorithm_name="ResNet", hyperparameters={}, status=TrialStatus.PRUNED, current_epoch=2),
            "trial_4": Trial(id="trial_4", algorithm_name="ResNet", hyperparameters={}, status=TrialStatus.ACTIVE, current_epoch=5, results={"accuracy": [(1, 0.6), (5, 0.85)]}),
            "trial_5": Trial(id="trial_5", algorithm_name="ResNet", hyperparameters={}, status=TrialStatus.PENDING),
        }
        experiment.trials = trials

        # 3. Create a scheduler
        scheduler = SuccessiveHalvingScheduler(metric="accuracy", increasing=True)

        # 4. Create the Runtime Engine
        # Callbacks can be dummy lambdas for this test
        runtime_engine = SdeRuntimeEngine(
            experiment=experiment,
            adaptive_scheduler=scheduler,
            trial_updated_callback=lambda x: None,
            insights_callback=lambda x: None,
            max_workers=1,
            enable_checkpointing=False,
        )

        # 5. Call the start method (which should trigger rehydration)
        runtime_engine.start(start_paused=True)

        # 6. Check the work queue
        work_queue = runtime_engine.work_queue
        self.assertEqual(work_queue.qsize(), 2)

        # Get the work units from the queue
        work_units = []
        while not work_queue.empty():
            work_units.append(work_queue.get()[2]) # [2] to get the WorkUnit object

        # Check that we have work units for the two active trials
        active_trial_ids = {"trial_2", "trial_4"}
        work_unit_trial_ids = {wu.trial_id for wu in work_units}
        self.assertEqual(active_trial_ids, work_unit_trial_ids)

        # Check that the work units are of the correct type
        for wu in work_units:
            self.assertEqual(wu.type, WorkUnitType.TRAIN_EPOCH)

        # Clean up the engine
        runtime_engine.stop()

if __name__ == "__main__":
    unittest.main()
