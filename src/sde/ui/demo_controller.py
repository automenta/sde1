import time
from typing import TYPE_CHECKING

from PyQt6.QtCore import QTimer

from .base_automation_controller import BaseAutomationController

if TYPE_CHECKING:
    from .components.spotlight_widget import SpotlightWidget
    from .main_window import MainWindow


class DemoController(BaseAutomationController):
    """Orchestrates the application demo, inheriting from BaseAutomationController."""

    def __init__(self, main_window: "MainWindow", spotlight: "SpotlightWidget"):
        super().__init__(main_window, update_interval=5000)
        self.spotlight = spotlight
        self.setup_pane = main_window.setup_pane
        self.results_pane = main_window.results_pane
        self._start_time = None

    def start(self):
        """Starts the demo sequence."""
        self._start_time = time.time()
        # 1. Select dataset
        self.setup_pane.dataset_combo.setCurrentText("MNIST")
        # 2. Select models
        model_list_widget = self.setup_pane.model_list
        for i in range(model_list_widget.count()):
            item = model_list_widget.item(i)
            if "SimpleCNN" in item.text() or "LogisticRegression" in item.text():
                item.setSelected(True)
        QTimer.singleShot(500, self.main_window.add_models_to_run)
        # 3. Start experiment
        QTimer.singleShot(
            1000,
            lambda: self.main_window.start_experiment(
                self.setup_pane.get_execution_settings()
            ),
        )
        # 4. Start the highlighting cycle
        super().start()

    def on_tick(self):
        """The main logic loop for the demo, called on each timer tick."""
        if time.time() - self._start_time > 60:  # Stop after 60 seconds
            self.main_window.stop_experiment()
            self.stop()
            self.spotlight.clear_highlight()
            return

        widgets = [
            self.results_pane.performance_plot,
            self.results_pane.trials_table_widget,
            self.results_pane.insights_widget,
            self.results_pane.log_widget,
        ]
        current_widget = self.spotlight.highlighted_widget()
        next_index = 0
        if current_widget in widgets:
            next_index = (widgets.index(current_widget) + 1) % len(widgets)
        self.spotlight.highlight(widgets[next_index])

    def stop(self):
        """Stops the demo sequence."""
        super().stop()
        self.spotlight.clear_highlight()