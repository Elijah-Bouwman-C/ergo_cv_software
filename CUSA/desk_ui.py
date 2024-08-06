import cv2
import mediapipe as mp
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QFileDialog

class InputDialog(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.layout = QVBoxLayout()

        self.video_path_label = QLabel('Select video file:', self)
        self.layout.addWidget(self.video_path_label)

        self.video_path_button = QPushButton('Browse', self)
        self.video_path_button.clicked.connect(self.browse_file)
        self.layout.addWidget(self.video_path_button)

        self.submit_button = QPushButton('Submit', self)
        self.submit_button.clicked.connect(self.submit)
        self.layout.addWidget(self.submit_button)

        self.setLayout(self.layout)

    def browse_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_path, _ = QFileDialog.getOpenFileName(self, "Select video file", "", "MP4 files (*.mp4 *.mov);;All files (*)", options=options)
        if file_path:
            self.video_path = file_path
            self.video_path_label.setText(f'Selected: {file_path}')

    def submit(self):
        self.close()

    def closeEvent(self, event):
        
        QApplication.instance().quit()

def get_height_weight_video_path():
    app = QApplication([])
    dialog = InputDialog()
    dialog.show()
    app.exec_()
    return dialog.video_path
