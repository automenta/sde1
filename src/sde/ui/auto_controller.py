import random
import itertools
from typing import TYPE_CHECKING, List, Set, Tuple, FrozenSet

from .base_automation_controller import BaseAutomationController
from .. import config
from ..registry import registry

if TYPE_CHECKING:
    from .main_window import MainWindow


class AutoController(BaseAutomationController):
    """
    A controller to automatically run experiments in a continuous loop,
    intelligently selecting novel combinations of challenges and models.
    """

    def __init__(self, main_window: "MainWindow"):
        update_interval_ms = config.AutoModeConfig.DELAY_BETWEEN_EXPERIMENTS_S * 1000
        super().__init__(main_window, update_interval=update_interval_ms)
        self.setup_pane = main_window.setup_pane
        self._experiment_history: Set[Tuple[str, FrozenSet[str]]] = set()
        self._all_possible_experiments = self._generate_all_possible_experiments()

    def start(self):
        """Starts the auto-discovery mode."""
        self._log_info("Auto-discovery mode started.")
        self.on_tick()  # Start the first experiment immediately
        super().start()

    def on_tick(self):
        """
        The main logic loop for auto-discovery. This method is called
        periodically to start a new experiment if one is not already running.
        """
        if self.view_model.experiment.status.is_running():
            self._log_info("Experiment is already running. Waiting for completion...")
            return

        self._log_info("Preparing next automated experiment...")
        self._run_new_experiment()

    def _generate_all_possible_experiments(self) -> List[Tuple[str, FrozenSet[str]]]:
        """Generates a list of all unique combinations of challenges and models."""
        all_challenges = registry.list_challenges()
        all_models = registry.list_models()
        experiments = []
        for challenge in all_challenges:
            for i in range(1, min(len(all_models), 3) + 1):
                for model_combo in itertools.combinations(all_models, i):
                    experiments.append((challenge, frozenset(model_combo)))
        random.shuffle(experiments)
        return experiments

    def _select_next_experiment(self) -> Tuple[str, List[str]]:
        """
        Selects a novel combination of challenge and models.
        If all combinations have been tried, it resets the history.
        """
        novel_experiments = [
            exp for exp in self._all_possible_experiments if exp not in self._experiment_history
        ]

        if not novel_experiments:
            self._log_info("All possible experiments have been run. Resetting and reshuffling.")
            self._experiment_history.clear()
            random.shuffle(self._all_possible_experiments)
            novel_experiments = self._all_possible_experiments

        challenge_name, selected_models_fs = novel_experiments[0]
        return challenge_name, list(selected_models_fs)

    def _run_new_experiment(self):
        """Selects and runs a new, preferably novel, experiment."""
        challenge_name, selected_models = self._select_next_experiment()

        self._log_info(f"Next up: Running {', '.join(selected_models)} on {challenge_name}.")

        # 1. Update UI to select the dataset (challenge)
        self.setup_pane.dataset_combo.setCurrentText(challenge_name)

        # 2. Update UI to select the models
        model_list_widget = self.setup_pane.model_list
        model_list_widget.clearSelection()
        for i in range(model_list_widget.count()):
            item = model_list_widget.item(i)
            if item.text() in selected_models:
                item.setSelected(True)

        # 3. Programmatically add models and start the experiment
        self.main_window.add_models_to_run()
        self.main_window.start_experiment(
            execution_settings=None,  # Use defaults
            patience_budget=config.AutoModeConfig.PATIENCE_BUDGET
        )

        # 4. Record the experiment in history
        history_entry = (challenge_name, frozenset(selected_models))
        self._experiment_history.add(history_entry)

    def _log_info(self, message: str):
        """Logs a message to the UI."""
        print(f"AutoController: {message}")
        self.main_window.append_log_message({"level": "INFO", "message": f"[Auto Mode] {message}"})

    def stop(self):
        """Stops the auto-discovery mode."""
        self._log_info("Auto-discovery mode stopped.")
        super().stop()