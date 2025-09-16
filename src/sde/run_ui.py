import multiprocessing
import sys

from PyQt6.QtWidgets import QApplication

from .discovery import discover_and_register_components
from .ui.main_window import MainWindow


def main():
    """The main entry point for the SDE application."""
    # --- Discover and register all components before creating the UI ---
    discover_and_register_components()

    app = QApplication(sys.argv)
    # It's good practice to set application-wide metadata
    app.setApplicationName("Scientific Discovery Engine")
    app.setOrganizationName("SDE-Project")

    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    # Set the start method to 'spawn' for CUDA compatibility on platforms
    # that default to 'fork'. This must be done within the __name__ == '__main__'
    # block and before any processes are created.
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        # This may be raised if the start method has already been set.
        pass
    main()
