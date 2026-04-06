import math
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QPen, QColor
from PyQt5.QtCore import Qt


class TiltIndicator(QWidget):
    _MARGIN = 12
    _TARGET_WIDTH = 2
    _CURRENT_WIDTH = 3
    _CURRENT_LENGTH = 0.6  # fraction of radius
    _MAX_ANGLE = 90.0

    _GREEN  = QColor(0, 220, 80)
    _YELLOW = QColor(230, 180, 0)
    _RED    = QColor(220, 50, 50)
    _WHITE  = QColor(255, 255, 255)
    _DIM    = QColor(80, 80, 80)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._target = 0.0
        self._current = 0.0
        self._active = True
        self._confidence = 1.0
        self.setMinimumSize(120, 120)

    # --- public API ---

    def set_target_tilt(self, angle: float):
        self._target = max(-self._MAX_ANGLE, min(self._MAX_ANGLE, angle))
        self.update()

    def set_current_tilt(self, angle: float):
        self._current = max(-self._MAX_ANGLE, min(self._MAX_ANGLE, angle))
        self.update()

    def set_assist_active(self, active: bool):
        self._active = active
        self.update()

    def set_confidence(self, conf: float):
        self._confidence = conf
        self.update()

    # --- rendering ---

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w, h = self.width(), self.height()
        cx, cy = w / 2, h / 2
        r = min(w, h) / 2 - self._MARGIN

        # circle border
        circle_color = self._DIM if not self._active else QColor(160, 160, 160)
        painter.setPen(QPen(circle_color, 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(int(cx - r), int(cy - r), int(r * 2), int(r * 2))

        # target line (full diameter, white)
        target_color = self._DIM if not self._active else self._WHITE
        self._draw_diameter(painter, cx, cy, r, self._target,
                            target_color, self._TARGET_WIDTH)

        # current line (short, from center outward, color-coded)
        current_color = self._current_color()
        self._draw_radius(painter, cx, cy, r * self._CURRENT_LENGTH,
                          self._current, current_color, self._CURRENT_WIDTH)

        painter.end()

    def _current_color(self):
        if not self._active:
            return self._DIM
        error = abs(self._current - self._target)
        if error < 3.0:
            return self._GREEN
        if error < 8.0:
            return self._YELLOW
        return self._RED

    @staticmethod
    def _angle_to_point(cx, cy, length, degrees):
        rad = math.radians(degrees)
        x = cx + length * math.cos(rad)
        y = cy - length * math.sin(rad)
        return x, y

    def _draw_diameter(self, painter, cx, cy, r, degrees, color, width):
        x1, y1 = self._angle_to_point(cx, cy, r, degrees)
        x2, y2 = self._angle_to_point(cx, cy, r, degrees + 180)
        painter.setPen(QPen(color, width))
        painter.drawLine(int(x1), int(y1), int(x2), int(y2))

    def _draw_radius(self, painter, cx, cy, length, degrees, color, width):
        x1, y1 = self._angle_to_point(cx, cy, length, degrees)
        painter.setPen(QPen(color, width))
        painter.drawLine(int(cx), int(cy), int(x1), int(y1))
