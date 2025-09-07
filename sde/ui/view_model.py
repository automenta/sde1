from datetime import datetime
from typing import Any
from typing import Dict
from typing import List

import pyqtgraph as pg
from PyQt6.QtCore import QObject
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QStyle
from sde.challenges import AVAILABLE_DATASETS
from sde.ui.models import UIAlgorithm
from sde.ui.models import UIInsight
from sde.ui.models import UITrial


class ExperimentViewModel(QObject):
    """Manages the UI state for the experiment.

    This class acts as a bridge between the backend `Orchestrator` and the
    `MainWindow`. It receives the raw state dictionary from the backend,
    converts it into UI-specific data models, and holds the complete state
    that the UI needs to render itself.
    """

    def __init__(self, style: QStyle, parent: QObject | None = None):
        super().__init__(parent)
        self._style = style
        self._raw_state: Dict[str, Any] = {}
        self.trials: Dict[str, UITrial] = {}
        self.algorithms: Dict[str, UIAlgorithm] = {}
        self.insights: List[UIInsight] = []
        self.status: str = "DEFINING"
        self.valid_actions: Dict[str, Any] = {}
        self.challenge_name: str = ""
        self.performance_metric_name: str = "accuracy"  # Default

        self._setup_icons()
        self._trial_plot_colors: Dict[str, pg.QtGui.QColor] = {}
        self._next_color_index = 0
        self.displayed_insight_messages = set()

    def update_state(self, new_state: Dict[str, Any]):
        """Updates the ViewModel with a new state dictionary from the backend."""
        self._raw_state = new_state
        self.status = new_state.get("status", "DEFINING")
        self.valid_actions = new_state.get("valid_actions", {})
        challenge_info = new_state.get("challenge", {})
        if challenge_info:
            self.challenge_name = challenge_info.get("name", "")

        self.algorithms = {
            id: UIAlgorithm(id=id, name=data["name"])
            for id, data in new_state.get("algorithms", {}).items()
        }
        self._update_trials(new_state.get("trials", {}))
        self._update_insights(new_state.get("insights", []))

    def _update_trials(self, trials_data: Dict[str, Any]):
        """Processes the raw trial data and updates the UI trial models."""
        best_trial_id = self._find_best_trial(trials_data)

        current_trial_ids = set(trials_data.keys())
        existing_trial_ids = set(self.trials.keys())

        # Remove trials that are no longer in the state
        for trial_id in existing_trial_ids - current_trial_ids:
            del self.trials[trial_id]
            if trial_id in self._trial_plot_colors:
                del self._trial_plot_colors[trial_id]

        # Create or update UITrial objects
        for trial_id, trial_data in trials_data.items():
            if trial_id not in self._trial_plot_colors:
                color = pg.intColor(self._next_color_index, hues=9, values=1)
                self._trial_plot_colors[trial_id] = color
                self._next_color_index += 1

            self.trials[trial_id] = UITrial(
                id=trial_data["id"],
                algorithm_name=trial_data["algorithm_name"],
                status=trial_data["status"],
                current_epoch=trial_data["current_epoch"],
                hyperparameters=trial_data["hyperparameters"],
                results=trial_data.get("results", {}),
                est_time_per_epoch=trial_data.get("est_time_per_epoch"),
                is_best=(trial_id == best_trial_id),
                plot_color=self._trial_plot_colors[trial_id],
            )

    def _find_best_trial(self, trials_data: Dict[str, Any]) -> str | None:
        """Determines the best trial based on the challenge's performance metric."""
        if not self.challenge_name or self.challenge_name not in AVAILABLE_DATASETS:
            return None

        challenge_def = AVAILABLE_DATASETS[self.challenge_name]
        metric_name = challenge_def.performance_metric_name
        self.performance_metric_name = metric_name
        higher_is_better = "accuracy" in metric_name.lower()

        best_trial_id = None
        best_perf = -float("inf") if higher_is_better else float("inf")

        for trial_id, trial_data in trials_data.items():
            results = trial_data.get("results", {})
            if metric_name in results and results[metric_name]:
                latest_perf = results[metric_name][-1][1]
                if (higher_is_better and latest_perf > best_perf) or (
                    not higher_is_better and latest_perf < best_perf
                ):
                    best_perf = latest_perf
                    best_trial_id = trial_id
        return best_trial_id

    def _update_insights(self, insights_data: List[Dict[str, Any]]):
        """Processes raw insight data and creates UI insight models."""
        for insight_data in insights_data:
            if insight_data["message"] not in self.displayed_insight_messages:
                self.displayed_insight_messages.add(insight_data["message"])
                self.insights.append(
                    UIInsight(
                        message=insight_data["message"],
                        type=insight_data["type"],
                        trial_ids=insight_data["trial_ids"],
                        timestamp=datetime.now().strftime("%H:%M:%S"),
                        icon=self.insight_icons.get(
                            insight_data["type"], self.insight_icons["DEFAULT"]
                        ),
                        raw_insight=insight_data,
                    )
                )

    def _setup_icons(self):
        """Pre-loads icons for different insight types."""
        self.insight_icons: Dict[str, QIcon] = {
            "BEST_PERFORMER": self._style.standardIcon(
                QStyle.StandardPixmap.SP_ArrowUp
            ),
            "PLATEAU": self._style.standardIcon(QStyle.StandardPixmap.SP_ArrowRight),
            "POOR_INITIAL_PERFORMANCE": self._style.standardIcon(
                QStyle.StandardPixmap.SP_MessageBoxWarning
            ),
            "PERFORMANCE_CROSSOVER": self._style.standardIcon(
                QStyle.StandardPixmap.SP_MediaSeekForward
            ),
            "HYPERPARAM_CORRELATION": self._style.standardIcon(
                QStyle.StandardPixmap.SP_DialogHelpButton
            ),
            "DEFAULT": self._style.standardIcon(
                QStyle.StandardPixmap.SP_MessageBoxInformation
            ),
        }

    def clear(self):
        """Resets the view model to a clean state for a new experiment."""
        self._raw_state = {}
        self.trials.clear()
        self.algorithms.clear()
        self.insights.clear()
        self.status = "DEFINING"
        self.valid_actions = {}
        self.challenge_name = ""
        self.performance_metric_name = "accuracy"
        self._trial_plot_colors.clear()
        self._next_color_index = 0
        self.displayed_insight_messages.clear()
