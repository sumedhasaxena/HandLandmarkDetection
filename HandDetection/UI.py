import sys
import os
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QPushButton, QFileDialog, QVBoxLayout, QLabel
from PyQt5.QtGui import QPixmap
from PyQt5 import uic

from landmark_detector import LandmarksDetector


class FileUploaderUI(QMainWindow):
    image_file_path = ""

    def __init__(self):
        super().__init__()

        uic.loadUi("Layout.ui", self)

        self.imageViewer.setScaledContents(True)

        self.open_file_dialog_button = self.findChild(QPushButton, "pushButton")
        self.run_detection_button = self.findChild(QPushButton, "pushButton_2")

        self.open_file_dialog_button.clicked.connect(self.open_file_dialog)
        self.run_detection_button.clicked.connect(self.run_detection)
        self.show()

    def open_file_dialog(self):
        file_dialog = QFileDialog()
        file_path = file_dialog.getOpenFileName(self, 'Select File')[0]
        if file_path:
            print(f"Selected file: {file_path}")
            self.image_file_path = file_path
            self.display_image(file_path)

    def display_image(self, file_path):
        pixmap = QPixmap(file_path)
        self.imageViewer.setPixmap(pixmap)

    def run_detection(self):
        detector = LandmarksDetector()
        img = detector.detect_and_draw_landmarks(self.image_file_path)
        file_name = os.path.basename(self.image_file_path)
        annotated_file_name = 'annotated_' + file_name
        img.save(annotated_file_name)
        self.display_image(annotated_file_name)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FileUploaderUI()
    # window.show()
    sys.exit(app.exec_())
