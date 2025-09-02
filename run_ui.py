import sys
from PyQt6.QtWidgets import QApplication
from sde.ui.main_window import MainWindow

def main():
    """
    The main entry point for the SDE application.
    """
    app = QApplication(sys.argv)
    # It's good practice to set application-wide metadata
    app.setApplicationName("Scientific Discovery Engine")
    app.setOrganizationName("SDE-Project")

    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
