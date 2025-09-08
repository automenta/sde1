import unittest

from sde.core.types import Experiment, Trial
from sde.core.types import TrialStatus
from sde.exploration.schedulers import HyperbandScheduler
from sde.exploration.schedulers import _Bracket


class TestHyperbandScheduler(unittest.TestCase):

    def test_bracket_initialization(self):
        """Tests that Hyperband initializes its brackets correctly."""
        # Setup: 13 trials to test assignment
        experiment = Experiment(id="test_exp")
        experiment.trials = {f"t{i}": Trial(id=f"t{i}", algorithm_name="algo", hyperparameters={}) for i in range(13)}

        # eta=3, max_resource=9 -> s_max=2. Brackets for s=2, 1, 0
        # Correct calculation for eta=3, max_resource=9 -> s_max=2:
        # s=2: n=ceil(3/3 * 3^2)=9. r=1. Trials: t0-t8
        # s=1: n=ceil(3/2 * 3^1)=5. r=3. Trials: t9-t12 (4 available)
        # s=0: n=ceil(3/1 * 3^0)=3. r=9. Trials: (0 available)
        scheduler = HyperbandScheduler(metric="acc", increasing=True, max_resource_per_trial=9, reduction_factor=3)

        work_units = scheduler.get_initial_work_units(experiment)

        self.assertEqual(len(scheduler.brackets), 3)

        # Check bracket s=2
        bracket_s2 = scheduler.brackets[0]
        self.assertEqual(bracket_s2.s, 2)
        self.assertEqual(bracket_s2.num_trials, 9)
        self.assertEqual(bracket_s2.rung_resources, [1, 3, 9])
        self.assertEqual(len(bracket_s2.trial_ids), 9)

        # Check bracket s=1
        bracket_s1 = scheduler.brackets[1]
        self.assertEqual(bracket_s1.s, 1)
        self.assertEqual(bracket_s1.num_trials, 5)
        self.assertEqual(bracket_s1.rung_resources, [3, 9])
        self.assertEqual(len(bracket_s1.trial_ids), 4) # Only 4 trials were left

        # Check bracket s=0
        bracket_s0 = scheduler.brackets[2]
        self.assertEqual(bracket_s0.s, 0)
        self.assertEqual(bracket_s0.num_trials, 3)
        self.assertEqual(bracket_s0.rung_resources, [9])
        self.assertEqual(len(bracket_s0.trial_ids), 0) # No trials were left

        # Check that all 13 trials were assigned and activated
        self.assertEqual(len(scheduler.trial_to_bracket), 13)
        self.assertEqual(len(work_units), 13)
        self.assertTrue(all(t.status == TrialStatus.ACTIVE for t in experiment.trials.values()))

    def test_pruning_logic_within_a_bracket(self):
        """Tests that the SHA logic correctly prunes trials within a single bracket."""
        # Setup: 5 trials to test a single bracket (s=1 from the example above)
        trials = {f"t{i}": Trial(id=f"t{i}", algorithm_name="algo", hyperparameters={}) for i in range(5)}

        # eta=3, max_resource=9
        scheduler = HyperbandScheduler(metric="acc", increasing=True, max_resource_per_trial=9, reduction_factor=3)

        # Manually set up the scheduler to simulate being in bracket s=1
        bracket_s1 = _Bracket(s=1, num_trials=5, rung_resources=[3, 9], trial_ids=[f"t{i}" for i in range(5)])
        scheduler.brackets.append(bracket_s1)
        for tid in bracket_s1.trial_ids:
            scheduler.trial_to_bracket[tid] = bracket_s1
            trials[tid].status = TrialStatus.ACTIVE

        # Simulate trials completing the first rung (3 epochs) with varying performance
        # t0: worst, t1: bad, t2: ok, t3: good, t4: best
        for i in range(5):
            trials[f"t{i}"].current_epoch = 3
            trials[f"t{i}"].results = {"acc": [(3, 0.5 + i*0.1)]}

        # The last trial (t4) finishes, triggering the pruning decision
        finished_trial = trials["t4"]
        new_work = scheduler.get_next_work_units(finished_trial, trials)

        # Check pruning: eta=3, 5 trials -> keep ceil(5/3) = 2 trials.
        # Survivors should be t4 and t3.
        # Pruned should be t0, t1, t2.
        self.assertEqual(trials["t4"].status, TrialStatus.ACTIVE)
        self.assertEqual(trials["t3"].status, TrialStatus.ACTIVE)
        self.assertEqual(trials["t2"].status, TrialStatus.PRUNED)
        self.assertEqual(trials["t1"].status, TrialStatus.PRUNED)
        self.assertEqual(trials["t0"].status, TrialStatus.PRUNED)

        # Check that the bracket's rung was advanced
        self.assertEqual(bracket_s1.rung, 1)

        # Check that new work was scheduled only for the 2 survivors
        self.assertEqual(len(new_work), 2)
        survivor_ids = {w.trial_id for w in new_work}
        self.assertEqual(survivor_ids, {"t3", "t4"})

    def test_pruning_with_missing_metric_in_bracket(self):
        """Test that a trial missing a metric is pruned correctly within a bracket."""
        trials = {f"t{i}": Trial(id=f"t{i}", algorithm_name="algo", hyperparameters={}) for i in range(3)}
        scheduler = HyperbandScheduler(metric="acc", increasing=True, max_resource_per_trial=3, reduction_factor=3)

        # Manually set up a bracket with 3 trials
        bracket = _Bracket(s=1, num_trials=3, rung_resources=[1, 3], trial_ids=[f"t{i}" for i in range(3)])
        scheduler.brackets.append(bracket)
        for tid in bracket.trial_ids:
            scheduler.trial_to_bracket[tid] = bracket
            trials[tid].status = TrialStatus.ACTIVE

        # Simulate completion of the first rung (1 epoch)
        trials["t0"].current_epoch = 1
        trials["t0"].results = {"acc": [(1, 0.9)]}
        trials["t1"].current_epoch = 1
        # t1 is missing its result
        trials["t2"].current_epoch = 1
        trials["t2"].results = {"acc": [(1, 0.8)]}

        # The last trial (t2) finishes, triggering the pruning decision
        finished_trial = trials["t2"]
        new_work = scheduler.get_next_work_units(finished_trial, trials)

        # Check pruning: eta=3, 3 trials -> keep ceil(3/3) = 1 trial.
        # Survivor should be t0. Pruned should be t1 and t2.
        self.assertEqual(trials["t0"].status, TrialStatus.ACTIVE)
        self.assertEqual(trials["t1"].status, TrialStatus.PRUNED)
        self.assertEqual(trials["t2"].status, TrialStatus.PRUNED)
        self.assertEqual(len(new_work), 1)
        self.assertEqual(new_work[0].trial_id, "t0")


if __name__ == '__main__':
    unittest.main()
