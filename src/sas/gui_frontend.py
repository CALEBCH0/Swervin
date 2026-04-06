import sys
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer

from sas.gui_client import Client
from sas.utils.tilt_indicator import TiltIndicator
from sas.utils.toml import load_toml, Config


def exception_hook(exctype, value, traceback):
    print(exctype, value, traceback)
    sys._excepthook(exctype, value, traceback)

sys._excepthook = sys.excepthook
sys.excepthook = exception_hook


class MainWindow(QWidget):
    def __init__(self, client: Client):
        super().__init__()
        self.client = client
        self.init_ui()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_view)
        self.timer.start(33)  # ~30 fps

    def init_ui(self):
        self.setWindowTitle("Swervin")
        self.setGeometry(100, 100, 960, 720)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.setLayout(layout)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.video_label, stretch=3)

        self.tilt_indicator = TiltIndicator()
        layout.addWidget(self.tilt_indicator, stretch=1)

    def update_view(self):
        frame = None
        while not self.client.frame_queue.empty():
            try:
                frame = self.client.frame_queue.get_nowait()
            except Exception:
                break

        if frame is not None:
            h, w, ch = frame.shape
            q_img = QImage(frame.data, w, h, 3 * w, QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(q_img).scaled(
                self.video_label.width(), self.video_label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.video_label.setPixmap(pixmap)

        label_data = None
        while not self.client.label_data_queue.empty():
            try:
                label_data = self.client.label_data_queue.get_nowait()
            except Exception:
                break

        if label_data is not None:
            self.tilt_indicator.set_current_tilt(label_data.get('current_tilt', 0.0))
            self.tilt_indicator.set_target_tilt(label_data.get('target_tilt', 0.0))

    def closeEvent(self, event):
        self.timer.stop()
        try:
            self.client.send_terminate()
        except Exception as e:
            print(f"Client: Error sending terminate signal: {e}")
        event.accept()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Q:
            self.close()
        elif event.key() == Qt.Key_R:
            self.client.send_start_rec()
        elif event.key() == Qt.Key_E:
            self.client.send_stop_rec()


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet("""
        QWidget { background-color: #222; color: #eee; }
    """)

    config = Config(load_toml("config.toml"))
    client = Client(server_ip=config.frontend.sock_host, port=config.frontend.sock_port)
    client.connect()
    client.start_receiving()

    window = MainWindow(client)
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
