from PyQt6.QtCore import QRect
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtGui import QPainter
from PyQt6.QtGui import QPen
from PyQt6.QtWidgets import QWidget


class SpotlightWidget(QWidget):
    """An overlay widget to highlight other widgets by dimming the background
    and drawing a border around the target widget.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        # Make the widget a borderless, transparent overlay that is always on top
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        # Let mouse events pass through to the widgets underneath
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._highlight_rect = QRect()
        self.hide()

    def paintEvent(self, event):
        """Paints the semi-transparent overlay and the highlight."""
        if self._highlight_rect.isNull() or not self.isVisible():
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Define the path for the entire widget area
        full_path = QPainterPath()
        full_path.addRect(self.rect())

        # Define the path for the highlighted (clear) area
        highlight_path = QPainterPath()
        highlight_path.addRoundedRect(self._highlight_rect, 5, 5)

        # Create a final path that is the difference between the full area and the highlight
        # This results in a "hole" in the overlay
        dimmed_path = full_path.subtracted(highlight_path)

        # Fill the dimmed area with a semi-transparent color
        painter.fillPath(dimmed_path, QColor(0, 0, 0, 128))

        # Draw a border around the highlighted area
        pen = QPen(QColor("#4A90E2"), 2)  # A nice blue color
        pen.setStyle(Qt.PenStyle.SolidLine)
        painter.setPen(pen)
        # Draw the rounded rectangle border, not a filled shape
        painter.drawPath(highlight_path)

    def highlight(self, widget: QWidget):
        """Calculates the position of the widget to highlight and triggers a repaint."""
        if not widget or not self.parentWidget():
            self.hide()
            return

        # The overlay should be the same size as its parent (the main window)
        self.setGeometry(self.parentWidget().rect())

        # Convert the widget's geometry to the global coordinate system,
        # then map it back to the overlay's coordinate system.
        target_pos = widget.mapToGlobal(widget.rect().topLeft())
        target_pos = self.mapFromGlobal(target_pos)
        self._highlight_rect = QRect(target_pos, widget.size()).adjusted(-5, -5, 5, 5)

        self.show()
        self.raise_()  # Ensure it's on top of other widgets
        self.update()  # Schedule a repaint

    def clear_highlight(self):
        """Hides the spotlight and clears the highlight rectangle."""
        self._highlight_rect = QRect()
        self.hide()
