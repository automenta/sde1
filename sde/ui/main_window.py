import sys
import uuid
import json
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTableWidget, QTableWidgetItem, QHeaderView, QTextEdit, QSplitter,
    QPushButton, QSizePolicy
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import pyqtgraph as pg

# Import backend components
from sde.engine.scheduler import Scheduler
from sde.core.types import Trial
from sde.exploration.schedulers import SuccessiveHalvingScheduler

class MainWindow(QMainWindow):
    """
    The main window for the Scientific Discovery Engine UI.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Scientific Discovery Engine")
        self.setGeometry(100, 100, 1200, 800)

        # --- Data maps for UI updates ---
        self.trial_row_map = {}  # trial.id -> table_row_index
        self.plot_curve_map = {} # trial.id -> plot_curve_item

        # --- Main Layout ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # --- Top Control Bar ---
        control_bar = QHBoxLayout()
        self.start_button = QPushButton("Start Experiment")
        self.start_button.clicked.connect(self.start_experiment)
        control_bar.addWidget(self.start_button)
        control_bar.addStretch(1)
        main_layout.addLayout(control_bar)

        # --- Main Content Splitter ---
        splitter = QSplitter(Qt.Orientation.Vertical)
        main_layout.addWidget(splitter)

        # --- Top Pane: Plot ---
        self.plot_widget = pg.PlotWidget()
        self.setup_plot()

        # --- Bottom Pane: Table and Log ---
        bottom_pane = QWidget()
        bottom_layout = QHBoxLayout(bottom_pane)

        self.trials_table = QTableWidget()
        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)

        bottom_splitter = QSplitter(Qt.Orientation.Horizontal)
        bottom_splitter.addWidget(self.trials_table)
        bottom_splitter.addWidget(self.log_text_edit)
        bottom_splitter.setSizes([750, 450])
        bottom_layout.addWidget(bottom_splitter)

        splitter.addWidget(self.plot_widget)
        splitter.addWidget(bottom_pane)
        splitter.setSizes([500, 300])

        self.setup_table()
        self.append_log_message("INFO: UI Initialized. Click 'Start Experiment' to begin.")

    def setup_plot(self):
        self.plot_widget.setBackground('w')
        self.plot_widget.setTitle("Real-Time Trial Performance", color="k", size="16pt")
        self.plot_widget.setLabel('left', 'Accuracy', color='k', **{'font-size': '12pt'})
        self.plot_widget.setLabel('bottom', 'Epoch', color='k', **{'font-size': '12pt'})
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.addLegend()

    def setup_table(self):
        self.trials_table.setColumnCount(6)
        self.trials_table.setHorizontalHeaderLabels([
            "Trial ID", "Status", "Epoch", "Accuracy", "Loss", "Hyperparameters"
        ])
        header = self.trials_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(True)
        self.trials_table.setColumnWidth(0, 120)
        self.trials_table.setColumnWidth(1, 100)
        self.trials_table.setColumnWidth(2, 80)
        self.trials_table.setColumnWidth(3, 100)
        self.trials_table.setColumnWidth(4, 100)

    def start_experiment(self):
        self.start_button.setEnabled(False)
        self.append_log_message("INFO: Starting new experiment...")

        # 1. Clear previous experiment data
        self.trials_table.setRowCount(0)
        self.plot_widget.clear()
        self.trial_row_map.clear()
        self.plot_curve_map.clear()
        self.setup_plot() # Re-add legend, etc.

        # 2. Define trials for the experiment
        # More trials to better showcase pruning
        trials = [
            Trial(
                id=f"t_{uuid.uuid4().hex[:6]}",
                algorithm_name="SimpleCNN",
                hyperparameters={'model_params': {'dropout_rate': r}, 'optimizer_params': {'lr': lr}}
            ) for r, lr in zip([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5],
                               [0.01, 0.05, 0.001, 0.005, 0.02, 0.002, 0.03, 0.003])
        ]

        # 3. Create the adaptive scheduler policy
        adaptive_scheduler = SuccessiveHalvingScheduler(
            metric="accuracy",
            increasing=True,
            min_epochs_per_rung=2,
            reduction_factor=2
        )

        # 4. Create and set up the main scheduler and worker thread
        self.scheduler_thread = QThread()
        self.scheduler = Scheduler(
            trials=trials,
            adaptive_scheduler=adaptive_scheduler,
            max_workers=4 # More workers for more trials
        )
        self.scheduler.moveToThread(self.scheduler_thread)

        # 4. Connect signals and slots
        self.scheduler.log_message.connect(self.append_log_message)
        self.scheduler.trial_updated.connect(self.update_trial_ui)
        self.scheduler.experiment_finished.connect(self.on_experiment_finished)
        self.scheduler.insight_generated.connect(self.append_log_message)

        # 5. Start the thread
        self.scheduler_thread.started.connect(self.scheduler.start)
        self.scheduler_thread.start()

    def update_trial_ui(self, trial_data: dict):
        trial_id = trial_data['id']

        # -- Update Table --
        if trial_id not in self.trial_row_map:
            row_position = self.trials_table.rowCount()
            self.trials_table.insertRow(row_position)
            self.trial_row_map[trial_id] = row_position
            # Add a basic plot curve for the new trial
            pen = pg.mkPen(color=(len(self.plot_curve_map) % 3 * 85, len(self.plot_curve_map) * 2 % 3 * 85, 255), width=2)
            self.plot_curve_map[trial_id] = self.plot_widget.plot(
                [], [], name=trial_id, pen=pen, symbol='o', symbolSize=6, symbolBrush=pen.color()
            )

        row = self.trial_row_map[trial_id]
        self.trials_table.setItem(row, 0, QTableWidgetItem(trial_id))
        self.trials_table.setItem(row, 1, QTableWidgetItem(trial_data['status']))
        self.trials_table.setItem(row, 2, QTableWidgetItem(str(trial_data['current_epoch'])))

        # Extract latest accuracy and loss if available
        accuracy_list = trial_data['results'].get('accuracy', [])
        loss_list = trial_data['results'].get('loss', [])
        latest_acc = f"{accuracy_list[-1][1]:.4f}" if accuracy_list else "N/A"
        latest_loss = f"{loss_list[-1][1]:.4f}" if loss_list else "N/A"

        self.trials_table.setItem(row, 3, QTableWidgetItem(latest_acc))
        self.trials_table.setItem(row, 4, QTableWidgetItem(latest_loss))
        self.trials_table.setItem(row, 5, QTableWidgetItem(json.dumps(trial_data['hyperparameters'])))

        # -- Update Plot --
        if accuracy_list:
            epochs = [item[0] for item in accuracy_list]
            accuracies = [item[1] for item in accuracy_list]
            self.plot_curve_map[trial_id].setData(epochs, accuracies)

    def append_log_message(self, message: str):
        self.log_text_edit.append(message)
        self.log_text_edit.verticalScrollBar().setValue(self.log_text_edit.verticalScrollBar().maximum())

    def on_experiment_finished(self):
        self.append_log_message("INFO: Experiment finished.")
        self.start_button.setEnabled(True)
        self.scheduler_thread.quit()
        self.scheduler_thread.wait()

    def closeEvent(self, event):
        """Ensure the scheduler thread is stopped cleanly on exit."""
        if hasattr(self, 'scheduler'):
            self.scheduler.stop()
        if hasattr(self, 'scheduler_thread') and self.scheduler_thread.isRunning():
            self.scheduler_thread.quit()
            self.scheduler_thread.wait()
        event.accept()


if __name__ == '__main__':
    # A simple test block to run and view the UI layout
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())
