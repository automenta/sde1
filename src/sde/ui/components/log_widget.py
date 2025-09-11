from datetime import datetime

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QGroupBox
from PyQt6.QtWidgets import QHBoxLayout
from PyQt6.QtWidgets import QPushButton
from PyQt6.QtWidgets import QTextEdit
from PyQt6.QtWidgets import QVBoxLayout
from PyQt6.QtWidgets import QWidget


class LogWidget(QWidget):
    """A widget for displaying log messages."""

    refresh_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        """Initializes the main UI layout and sub-components."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        log_group = QGroupBox("Log")
        log_outer_layout = QVBoxLayout()
        log_group.setLayout(log_outer_layout)

        log_header_layout = QHBoxLayout()
        log_header_layout.addStretch()
        self.refresh_button = QPushButton("Refresh")
        self.clear_log_button = QPushButton("Clear Log")
        log_header_layout.addWidget(self.refresh_button)
        log_header_layout.addWidget(self.clear_log_button)

        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)

        log_outer_layout.addLayout(log_header_layout)
        log_outer_layout.addWidget(self.log_text_edit)
        log_outer_layout.setContentsMargins(0, 5, 0, 0)

        main_layout.addWidget(log_group)

    def _connect_signals(self):
        """Connects internal widget signals."""
        self.clear_log_button.clicked.connect(self.clear_log)
        self.refresh_button.clicked.connect(self.refresh_requested)

    def append_log_message(self, log_data: dict):
        """Appends a structured log message to the text edit with color-coding."""
        level = log_data.get("level", "INFO").upper()
        message = log_data.get("message", "")

        color_map = {
            "INFO": "#000000",      # Black
            "WARN": "#FFA500",      # Orange
            "ERROR": "#DC143C",     # Crimson
            "INSIGHT": "#8A2BE2",   # BlueViolet
        }
        color = color_map.get(level, "black")

        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = (
            f'<span style="color: #808080;">[{timestamp}]</span> '
            f'<b style="color: {color};">[{level}]</b> '
            f'<span style="color: #36454F;">{message}</span>'
        )
        self.log_text_edit.append(formatted_message)
        self.log_text_edit.verticalScrollBar().setValue(
            self.log_text_edit.verticalScrollBar().maximum()
        )

    def clear_log(self):
        """Clears the log text edit."""
        self.log_text_edit.clear()
        self.append_log_message({"level": "INFO", "message": "Log cleared."})
