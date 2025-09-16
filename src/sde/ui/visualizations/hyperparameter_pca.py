from typing import Optional

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QVBoxLayout
from PyQt6.QtWidgets import QWidget
from sklearn.decomposition import PCA

from ..view_model import ExperimentViewModel
from .base import VisualizationPlugin


class HyperparameterPCAWidget(QWidget):
    """A widget for visualizing hyperparameter PCA."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.view_model: Optional[ExperimentViewModel] = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setTitle("Hyperparameter PCA")
        self.plot_widget.setLabel("left", "Principal Component 2")
        self.plot_widget.setLabel("bottom", "Principal Component 1")
        layout.addWidget(self.plot_widget)

    def set_view_model(self, view_model: ExperimentViewModel):
        self.view_model = view_model

    def update_plots(self, trials):
        if not self.view_model or not trials:
            return

        hparams = []
        for trial in trials.values():
            hparams.append(list(trial.hyperparameters.values()))

        if len(hparams) < 2:
            return

        hparams = np.array(hparams)

        # very basic PCA for now
        try:
            pca = PCA(n_components=2)
            transformed_hparams = pca.fit_transform(hparams)

            self.plot_widget.clear()
            scatter = pg.ScatterPlotItem(
                x=transformed_hparams[:, 0],
                y=transformed_hparams[:, 1],
                size=10,
            )
            self.plot_widget.addItem(scatter)
        except Exception as e:
            print(f"PCA Error: {e}")

    def clear(self):
        self.plot_widget.clear()


class HyperparameterPCAPlugin(VisualizationPlugin):
    """Plugin for the hyperparameter PCA plot."""

    @property
    def name(self) -> str:
        return "Hyperparameter PCA"

    def create_widget(
        self, parent: Optional[QWidget], view_model: ExperimentViewModel
    ) -> QWidget:
        widget = HyperparameterPCAWidget(parent)
        widget.set_view_model(view_model)
        return widget
