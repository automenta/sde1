import random
from collections.abc import Callable
from typing import TYPE_CHECKING

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QListView
from PyQt6.QtWidgets import QWidget

if TYPE_CHECKING:
    from .components.spotlight_widget import SpotlightWidget
    from .main_window import MainWindow


class DemoController:
    """Orchestrates the application demo, simulating user interaction to showcase
    features in an automated and educational tour.
    """

    def __init__(self, main_window: "MainWindow", spotlight: "SpotlightWidget"):
        self.main_window = main_window
        self.spotlight = spotlight
        self.view_model = main_window.view_model
        self.setup_pane = main_window.setup_pane
        self.results_pane = main_window.results_pane

        self._demo_timer = QTimer(main_window)
        self._demo_timer.timeout.connect(self._execute_next_step)
        self._demo_steps: list[dict] = []
        self._current_step = 0

    def start(self):
        """Starts the demo sequence."""
        self._prepare_demo_steps()
        self._current_step = 0
        self._execute_next_step()

    def _prepare_demo_steps(self):
        """Defines the sequence of actions for the demo."""
        self._demo_steps = [
            {
                "func": self._step_select_dataset,
                "widget": self.setup_pane.dataset_combo,
                "text": "Welcome to the SDE Demo! First, let's select a dataset. We'll use MNIST.",
                "delay": 2000,
            },
            {
                "func": self._step_select_models,
                "widget": self.setup_pane.model_list,
                "text": "Now, let's pick some models to compare. We'll try a few classical and vision models.",
                "delay": 2000,
            },
            {
                "func": self._step_start_experiment,
                "widget": self.setup_pane.start_button,
                "text": "With everything configured, let's start the experiment!",
                "delay": 3000,
            },
            {
                "func": self._step_let_it_run_1,
                "widget": self.results_pane.log_widget,
                "text": "The experiment is now running. The system is training and evaluating different model configurations, called 'trials', in the background.",
                "delay": 8000,
            },
            {
                "func": self._step_select_a_trial,
                "widget": self.results_pane.trials_table_widget,
                "text": "We can select a trial in the table to see its performance curve highlighted in the plot.",
                "delay": 3000,
            },
            {
                "func": self._step_select_an_insight,
                "widget": self.results_pane.insights_widget,
                "text": "The system generates insights, like identifying the best performer. Clicking an insight highlights the relevant trials.",
                "delay": 4000,
            },
            {
                "func": self._step_pause_experiment,
                "widget": self.setup_pane.pause_button,
                "text": "Let's pause the run. No new trials will be started.",
                "delay": 3000,
            },
            {
                "func": self._step_resume_experiment,
                "widget": self.setup_pane.resume_button,
                "text": "And now, let's resume the experiment.",
                "delay": 5000,
            },
            {
                "func": self._step_let_it_run_2,
                "widget": self.results_pane.performance_plot,
                "text": "The experiment is running again. Notice how the plot and trial table update in real-time.",
                "delay": 8000,
            },
            {
                "func": self._step_double_click_trial,
                "widget": self.results_pane.trials_table_widget,
                "text": "Double-clicking a trial shows its exact hyperparameters.",
                "delay": 3000,
            },
            {
                "func": self._step_stop_experiment,
                "widget": self.setup_pane.stop_button,
                "text": "After gathering enough data, we can stop the run. This will terminate all workers.",
                "delay": 3000,
            },
            {
                "func": self._step_finished,
                "widget": None,
                "text": "Demo finished! You can now explore the results or close the application.",
                "delay": 0,
            },
        ]

    def _execute_next_step(self):
        """Executes the current demo step and schedules the next one."""
        if self._current_step > 0:
            self._cleanup_step()

        if self._current_step >= len(self._demo_steps):
            return

        step_data = self._demo_steps[self._current_step]

        self.main_window.append_log_message({
            "level": "DEMO",
            "message": step_data["text"],
        })

        if step_data["widget"]:
            self.spotlight.highlight(step_data["widget"])

        step_data["func"]()

        self._current_step += 1

        if step_data["delay"] > 0:
            self._demo_timer.start(step_data["delay"])

    def _cleanup_step(self):
        """Clears artifacts from the previous step, like highlights."""
        self.spotlight.clear_highlight()

    # --- Demo Steps ---

    def _step_select_dataset(self):
        self.setup_pane.dataset_combo.setCurrentText("MNIST")

    def _step_select_models(self):
        model_list_widget = self.setup_pane.model_list
        num_models = model_list_widget.count()
        if num_models > 0:
            num_to_select = min(3, num_models)
            model_indices = random.sample(range(num_models), num_to_select)
            for index in model_indices:
                item = model_list_widget.item(index)
                item.setSelected(True)
        QTimer.singleShot(500, self.main_window.add_models_to_run)

    def _step_start_experiment(self):
        self.main_window.start_experiment(self.setup_pane.get_execution_settings())

    def _step_let_it_run_1(self):
        pass

    def _step_select_a_trial(self):
        if self.results_pane.trials_table_widget.rowCount() > 0:
            self.results_pane.trials_table_widget.selectRow(0)

    def _step_select_an_insight(self):
        if self.results_pane.insights_widget.insights_list.count() > 0:
            item = self.results_pane.insights_widget.insights_list.item(0)
            if item:
                self.results_pane.insights_widget.insights_list.setCurrentItem(item)
                self.results_pane.insights_widget._on_item_clicked(item)

    def _step_pause_experiment(self):
        self.main_window.pause_experiment()

    def _step_resume_experiment(self):
        self.main_window.resume_experiment()

    def _step_let_it_run_2(self):
        pass

    def _step_double_click_trial(self):
        if self.results_pane.trials_table_widget.rowCount() > 0:
            trial_id_item = self.results_pane.trials_table_widget.item(0, 0)
            if trial_id_item:
                trial_id = trial_id_item.text()
                self.main_window.on_trial_double_clicked(trial_id)

    def _step_stop_experiment(self):
        self.main_window.stop_experiment()

    def _step_finished(self):
        self.spotlight.clear_highlight()
        self._demo_timer.stop()
