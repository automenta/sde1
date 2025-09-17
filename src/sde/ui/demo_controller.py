import time
from typing import TYPE_CHECKING

from PyQt6.QtCore import QTimer

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
        self._demo_timer = None
        self._start_time = None

    def start(self):
        """Starts the demo sequence."""
        self.run_continuous_demo()

    def run_continuous_demo(self):
        """Runs a continuous, hands-off demo of the application."""
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
        # 4. Dynamically highlight UI elements
        self._demo_timer = QTimer(self.main_window)
        self._demo_timer.timeout.connect(self._highlight_next_widget)
        self._demo_timer.start(5000)

    def _highlight_next_widget(self):
        """Cycles through highlighting different widgets."""
        if time.time() - self._start_time > 60:  # Stop after 60 seconds
            self.main_window.stop_experiment()
            self._demo_timer.stop()
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
