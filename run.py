from EJMS import main_ejms as lifting_tool
from CUSA import desk_assess as cusa_tool
from Hand_Tools import main_hand_analysis as hand_tool

import sys
from random import randint

from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QRadioButton,
    QLineEdit,
    QFileDialog,
    )


class hand_window(QWidget):

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.video_path_label = QLabel('Select video file:', self)
        layout.addWidget(self.video_path_label)
        self.video_path_button = QPushButton('Browse', self)
        self.video_path_button.clicked.connect(self.browse_file)
        layout.addWidget(self.video_path_button)
        self.submit_button = QPushButton('Submit', self)
        self.submit_button.clicked.connect(self.submit)
        layout.addWidget(self.submit_button)
        self.setLayout(layout)
    
    def browse_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_path, _ = QFileDialog.getOpenFileName(self, "Select video file", "", "Video files (*.mp4 *.mov);;All files (*)", options=options)
        try:
            if file_path:
                self.video_path = file_path
                self.video_path_label.setText(f'Selected: {file_path}')
        except Exception as e:
            print(e)

    def submit(self):
        try:
            hand_tool.main(self.video_path)
            self.close()
        except Exception as e:
            print(e)
            self.submit



class cusa_window(QWidget):

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.video_path_label = QLabel('Select video file:', self)
        layout.addWidget(self.video_path_label)
        self.video_path_button = QPushButton('Browse', self)
        self.video_path_button.clicked.connect(self.browse_file)
        layout.addWidget(self.video_path_button)
        self.submit_button = QPushButton('Submit', self)
        self.submit_button.clicked.connect(self.submit)
        layout.addWidget(self.submit_button)
        self.setLayout(layout)
    
    def browse_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_path, _ = QFileDialog.getOpenFileName(self, "Select video file", "", "Video files (*.mp4 *.mov);;All files (*)", options=options)
        try:
            if file_path:
                self.video_path = file_path
                self.video_path_label.setText(f'Selected: {file_path}')
        except Exception as e:
            print(e)

    def submit(self):
        try:
            cusa_tool.main(self.video_path)
            self.close()
        except Exception as e:
            print(e)
            self.submit



class lift_window(QWidget):

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        
        self.height_label = QLabel('Enter your height (in ft):', self)
        layout.addWidget(self.height_label)
        
        self.height_input = QLineEdit(self)
        layout.addWidget(self.height_input)

        self.weight_label = QLabel('Enter your weight (in lbs):', self)
        layout.addWidget(self.weight_label)

        self.weight_input = QLineEdit(self)
        layout.addWidget(self.weight_input)

        self.video_path_label = QLabel('Select video file:', self)
        layout.addWidget(self.video_path_label)

        self.video_path_button = QPushButton('Browse', self)
        self.video_path_button.clicked.connect(self.browse_file)
        layout.addWidget(self.video_path_button)

        self.submit_button = QPushButton('Submit', self)
        self.submit_button.clicked.connect(self.submit)
        layout.addWidget(self.submit_button)

        self.setLayout(layout)

    def browse_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_path, _ = QFileDialog.getOpenFileName(self, "Select video file", "", "Video files (*.mp4 *.mov);;All files (*)", options=options)
        try:
            if file_path:
                self.video_path = file_path
                self.video_path_label.setText(f'Selected: {file_path}')
        except Exception as e:
            print(e)

    def submit(self):
        try:
            self.height = float(self.height_input.text())
            self.weight = float(self.weight_input.text())
            lifting_tool.main(self.height,self.weight,self.video_path)
            self.close()
        except Exception as e:
            print(e)
            self.submit



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.window1 = lift_window()
        self.window2 = hand_window()
        self.window3 = cusa_window()
        l = QVBoxLayout()
        button1 = QPushButton("Lifting/pushing/pulling Assessment")
        button1.clicked.connect(
            lambda checked: self.toggle_window(self.window1)
        )
        l.addWidget(button1)

        button2 = QPushButton("Hand Assessment")
        button2.clicked.connect(
            lambda checked: self.toggle_window(self.window2)
        )
        l.addWidget(button2)
        
        button3 = QPushButton("CUSA Assessment")
        button3.clicked.connect(
            lambda checked: self.toggle_window(self.window3)
        )
        l.addWidget(button3)
        
        w = QWidget()
        w.setLayout(l)
        self.setCentralWidget(w)

    def toggle_window(self, window):
        if window.isVisible():
            window.hide()

        else:
            window.show()


app = QApplication(sys.argv)
w = MainWindow()
w.show()
app.exec()
