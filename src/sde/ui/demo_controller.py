import random
from collections.abc import Callable
from typing import TYPE_CHECKING

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QListView

if TYPE_CHECKING:
    from .main_window import MainWindow


class DemoController:
    """Orchestrates the application demo, simulating user interaction to showcase
    features in an automated and educational tour.
    """

    def __init__(self, main_window: "MainWindow"):
        self.main_window = main_window
        self.view_model = main_window.view_model
        self.setup_pane = main_window.setup_pane
        self.results_pane = main_window.results_pane

        self._demo_timer = QTimer(main_window)
        self._demo_timer.timeout.connect(self._execute_next_step)
        self._demo_steps: list[tuple[Callable, int]] = []
        self._current_step = 0

    def start(self):
        """Starts the demo sequence."""
        self._prepare_demo_steps()
        self._current_step = 0
        self._execute_next_step()

    def _prepare_demo_steps(self):
        """Defines the sequence of actions for the demo."""
        self._demo_steps = [
            (self._step_select_dataset, 2000),
            (self._step_select_models, 2000),
            (self._step_start_experiment, 3000),
            (self._step_let_it_run_1, 8000),
            (self._step_select_a_trial, 3000),
            (self._step_select_an_insight, 4000),
            (self._step_pause_experiment, 3000),
            (self._step_resume_experiment, 5000),
            (self._step_let_it_run_2, 8000),
            (self._step_double_click_trial, 3000),
            (self._step_stop_experiment, 3000),
            (self._step_finished, 0),
        ]

    def _execute_next_step(self):
        """Executes the current demo step and schedules the next one."""
        if self._current_step >= len(self._demo_steps):
            return

        step_func, delay = self._demo_steps[self._current_step]
        step_func()
        self._current_step += 1

        if self._current_step < len(self._demo_steps):
            next_delay = self._demo_steps[self._current_step][1]
            if next_delay > 0:
                self._demo_timer.start(next_delay)

    # --- Demo Steps ---

    def _step_select_dataset(self):
        """Selects the MNIST dataset."""
        self.main_window.append_log_message({
            "level": "DEMO",
            "message": "Welcome to the SDE Demo! First, let's select a dataset. We'll use MNIST."
        })
        self.setup_pane.dataset_combo.setCurrentText("MNIST")

    def _step_select_models(self):
        """Selects a few models to run."""
        self.main_window.append_log_message({
            "level": "DEMO",
            "message": "Now, let's pick some models to compare. We'll try a few classical and vision models."
        })
        model_list_widget = self.setup_pane.model_list
        num_models = model_list_widget.count()
        if num_models > 0:
            # Select up to 3 random models
            num_to_select = min(3, num_models)
            model_indices = random.sample(range(num_models), num_to_select)
            for index in model_indices:
                item = model_list_widget.item(index)
                item.setSelected(True)

        # A small delay to make the selection visible before adding
        QTimer.singleShot(500, self.main_window.add_models_to_run)


    def _step_start_experiment(self):
        """Clicks the 'Start' button."""
        self.main_window.append_log_message({
            "level": "DEMO",
            "message": "With everything configured, let's start the experiment!"
        })
        self.main_window.start_experiment(self.setup_pane.get_execution_settings())

    def _step_let_it_run_1(self):
        self.main_window.append_log_message({
            "level": "DEMO",
            "message": "The experiment is now running. Workers are processing trials in the background."
        })

    def _step_select_a_trial(self):
        """Selects the first available trial in the table."""
        self.main_window.append_log_message({
            "level": "DEMO",
            "message": "We can select a trial in the table to see its performance curve highlighted in the plot."
        })
        if self.results_pane.trials_table_widget.rowCount() > 0:
            self.results_pane.trials_table_widget.selectRow(0)

    def _step_select_an_insight(self):
        """Selects the first available insight."""
        self.main_window.append_log_message({
            "level": "DEMO",
            "message": "The system generates insights, like identifying the best performer. Clicking an insight highlights the relevant trials."
        })
        if self.results_pane.insights_widget.insights_list.count() > 0:
            item = self.results_pane.insights_widget.insights_list.item(0)
            if item:
                self.results_pane.insights_widget.insights_list.setCurrentItem(item)
                self.results_pane.insights_widget._on_item_clicked(item)

    def _step_pause_experiment(self):
        """Pauses the experiment."""
        self.main_window.append_log_message({
            "level": "DEMO",
            "message": "Let's pause the run. No new trials will be started."
        })
        self.main_window.pause_experiment()

    def _step_resume_experiment(self):
        """Resumes the experiment."""
        self.main_window.append_log_message({
            "level": "DEMO",
            "message": "And now, let's resume the experiment."
        })
        self.main_window.resume_experiment()

    def _step_let_it_run_2(self):
        self.main_window.append_log_message({
            "level": "DEMO",
            "message": "The experiment is running again. Notice how the plot and trial table update in real-time."
        })

    def _step_double_click_trial(self):
        """Double-clicks a trial to show its hyperparameters."""
        self.main_window.append_log_message({
            "level": "DEMO",
            "message": "Double-clicking a trial shows its exact hyperparameters."
        })
        if self.results_pane.trials_table_widget.rowCount() > 0:
            # Get the trial ID from the first row
            trial_id_item = self.results_pane.trials_table_widget.item(0, 0)
            if trial_id_item:
                trial_id = trial_id_item.text()
                self.main_window.on_trial_double_clicked(trial_id)
                # Close the dialog after a short delay
                QTimer.singleShot(2000, self.main_window.dialog_service.parent.findChild(QDialog).close)

    def _step_stop_experiment(self):
        """Stops the experiment."""
        self.main_window.append_log_message({
            "level": "DEMO",
            "message": "After gathering enough data, we can stop the run. This will terminate all workers."
        })
        self.main_window.stop_experiment()

    def _step_finished(self):
        """Indicates the demo is finished."""
        self.main_window.append_log_message({
            "level": "DEMO",
            "message": "Demo finished! You can now explore the results or close the application."
        })
        self._demo_timer.stop()
