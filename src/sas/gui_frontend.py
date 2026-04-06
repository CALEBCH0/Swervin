import sys
import math
from time import monotonic
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
)
from PyQt5.QtGui import (
    QPixmap, QImage, QColor, QPainter, QPen, QFont
)
from PyQt5.QtCore import Qt, QTimer, QRect
import numpy as np
import threading
import cv2
import random
import pyqtgraph as pg
from sas.utils.toml import load_toml
from sas.gui_client import Client

def exception_hook(exctype, value, traceback):
    print(exctype, value, traceback)
    sys._excepthook(exctype, value, traceback)

sys._excepthook = sys.excepthook
sys.excepthook = exception_hook

class MainWindow(QWidget):
    def __init__(self, app_config, client):
        super().__init__()
        self.app_config = app_config
        self.client = client
        self.init_ui()
        self.start_time = monotonic()
        self.frame_count = 0

    def init_ui(self):
        self.setWindowTitle("SAS Visualization")
        self.setGeometry(100, 100, self.app_config.frame_width, self.app_config.frame_height)

        # Main layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Video display label
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.video_label)

    def update_frame(self, frame, sas_status):
        # Convert the frame to QImage and display it
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        
        # Draw SAS status on the image
        painter = QPainter(q_img)
        pen = QPen(QColor(255, 0, 0), 2)
        painter.setPen(pen)
        font = QFont('Arial', 16)
        painter.setFont(font)
        
        # Example: Draw lane departure warning
        if sas_status.get('lane_departure_warning', False):
            painter.drawText(10, 30, "Lane Departure Warning!")
        
        painter.end()

        pixmap = QPixmap.fromImage(q_img)
        self.video_label.setPixmap(pixmap)

    def record_frame(self):
        # Capture the screen of the whole window
        screen = QApplication.primaryScreen()
        window_image = screen.grabWindow(self.winId())

        # Convert QImage to a format OpenCV can use
        frame = window_image.toImage()
        frame = frame.convertToFormat(4)  # Format_ARGB32

        width = frame.width()
        height = frame.height()
        ptr = frame.bits()
        ptr.setsize(frame.byteCount())
        arr = np.array(ptr).reshape(height, width, 4)  # Copies the data

        # Convert RGBA to BGR for OpenCV
        rgb_frame = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
        #bgr_frame = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)

        # Save each frame as a PNG image
        filename = f"frame_{self.frame_counter:04d}.png"
        cv2.imwrite(filename, rgb_frame)

        print(f"Saved {filename}")
        self.frame_counter += 1

    def update_view(self):
        pass

    def closeEvent(self, event):
        """Handle window close event"""
        print("Client: Terminating...")
        try:
            if hasattr(self, "client"):
                self.client.send_terminate()
        except Exception as e:
            print(f"Client: Error sending terminate signal: {e}")
        
        self.timer.stop()
        event.accept()

    def keyPressedEvent(self, event):
        """Handle key press events and send commands to the server"""
        if event.key() == Qt.Key_Q:
            self.close()
        elif event.key() == Qt.Key_R:
            print("Client: Sending record command")
            self.client.send_start_rec()
        elif event.key() == Qt.Key_E:
            print("Client: Sending stop recording command")
            self.client.send_stop_rec()


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet("""
        QWidget {
            background-color: #222;
            color: #eee;
        }
        QLabel {
            font-size: 14px;
        }
    """)

    fw, fh = [(1920, 1080), (1080, 720)][0] # TODO: currently set for 1080p camera input
    client = Client(server_ip=app_config.host_ip, frame_width=fw, frame_height=fh)
    client.connect()
    use = [""]

    window = MainWindow(client, use)

    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()