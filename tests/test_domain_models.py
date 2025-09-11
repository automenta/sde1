import uuid
from sde.core.domain import (
    AlgorithmConfig,
    DatasetType,
    ExecutionSettings,
    Experiment,
    ExperimentStatus,
    Trial,
    TrialStatus,
)


def test_experiment_serialization_roundtrip():
    """
    Tests that an Experiment object can be serialized to a dictionary
    and then deserialized back into an identical object.
    This verifies the to_dict and from_dict methods.
    """
    # 1. Create a complex Experiment object
    original_settings = ExecutionSettings(
        num_trials_per_algo=10,
        num_workers=4,
        enable_checkpointing=True,
        work_unit_timeout_seconds=300,
    )

    original_algo = AlgorithmConfig(
        id="algo_1",
        name="TestAlgo",
        parameter_space={"lr": (0.001, 0.1)},
        is_active=True,
    )

    original_trial = Trial(
        id="trial_1",
        algorithm_name="TestAlgo",
        hyperparameters={"lr": 0.05},
        status=TrialStatus.ACTIVE,
        priority=10,
        current_epoch=5,
        checkpoint_path="/tmp/checkpoint.pt",
        est_time_per_epoch=1.2,
        results={"accuracy": [(1, 0.5), (2, 0.6)]},
    )

    original_experiment = Experiment(
        id=f"exp_{uuid.uuid4().hex[:8]}",
        status=ExperimentStatus.RUNNING,
        challenge={"name": "CIFAR10", "type": DatasetType.IMAGE_CLASSIFICATION},
        algorithms={"algo_1": original_algo},
        trials={"trial_1": original_trial},
        insights=[{"type": "TestInsight", "content": "Test"}],
        adaptive_policy="Hyperband",
        patience_budget={"max_epochs": 100},
        execution_settings=original_settings,
        scheduler_state={"rung": 1},
    )

    # 2. Serialize to dictionary
    experiment_dict = original_experiment.to_dict()

    # 3. Deserialize back to an object
    recreated_experiment = Experiment.from_dict(experiment_dict)

    # 4. Assert that the recreated object is identical to the original
    # The default dataclass __eq__ does a field-by-field comparison.
    assert original_experiment == recreated_experiment


def test_from_dict_robustness():
    """
    Tests that the from_dict methods can handle extra, unknown keys
    in the input dictionary, which provides forward compatibility.
    """
    trial_dict_with_extra_keys = {
        "id": "trial_2",
        "algorithm_name": "FutureAlgo",
        "hyperparameters": {"bs": 64},
        "status": "ACTIVE",
        "some_future_field": "some_value",
        "another_new_property": [1, 2, 3],
    }

    # This should not raise an error
    trial = Trial.from_dict(trial_dict_with_extra_keys)

    assert trial.id == "trial_2"
    assert trial.status == TrialStatus.ACTIVE
    # Assert that the unknown fields were ignored and not added to the object
    assert not hasattr(trial, "some_future_field")
