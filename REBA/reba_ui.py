import cv2
import mediapipe as mp
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QFileDialog
import os 

class InputDialog(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.layout = QVBoxLayout()

        self.height_label = QLabel('Enter your height (in ft):', self)
        self.layout.addWidget(self.height_label)
        
        self.height_input = QLineEdit(self)
        self.layout.addWidget(self.height_input)

        self.weight_label = QLabel('Enter your weight (in lbs):', self)
        self.layout.addWidget(self.weight_label)

        self.weight_input = QLineEdit(self)
        self.layout.addWidget(self.weight_input)
        
        self.coupling_label = QLabel('Enter the coupling (good,fair,poor,unacceptable)', self)
        self.layout.addWidget(self.coupling_label)

        self.coupling_const = QLineEdit(self)
        self.layout.addWidget(self.coupling_const)
        
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
        file_path, _ = QFileDialog.getOpenFileName(self, "Select video file", "", "Video files (*.mp4 *.mov);;All files (*)", options=options)
        try:
            if file_path:
                self.video_path = file_path
                self.video_path_label.setText(f'Selected: {file_path}')
        except:
            self.file_error_label = QLabel('Filepath invalid please try again', self)
            self.layout.addWidget(self.file_error_label)
            self.browse_file

    def submit(self):
        # try:
            self.height = float(self.height_input.text())
            self.weight = float(self.weight_input.text())
            self.coupling_const = str(self.coupling_const.text())
            if self.coupling_const not in {'good','fair','poor','unacceptable'}:
                self.error_label = QLabel('Weight or height invalid please try again', self)
                self.layout.addWidget(self.error_label)    
                self.submit
            self.close()
        # except:
        #     self.error_label = QLabel('Weight or height invalid please try again', self)
        #     self.layout.addWidget(self.error_label)
        #     self.submit

    def closeEvent(self, event):
        
        QApplication.instance().quit()

def get_height_weight_video_path():
    app = QApplication([])
    dialog = InputDialog()
    dialog.show()
    app.exec_()
    return dialog.height, dialog.weight, dialog.video_path, dialog.coupling_const

