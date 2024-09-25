# import do_kilosort as k
# import os
# import shared.config as config
# import numpy as np
# import sys
# from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QProgressBar, QPushButton, QLabel
# from PyQt5.QtGui import QPalette, QColor

# class MainWindow(QMainWindow):
#     def __init__(self):
#         super(MainWindow, self).__init__()
#         self.setWindowTitle("Spike Sorting Control Loop GUI")
        
#         #set up layouts
#         self.setAutoFillBackground(True)
#         HorizontalLayout = QHBoxLayout()
        
#         SortingLayout = QVBoxLayout()
#         TemplateChoosingLayout = QVBoxLayout()
#         BonsaiLayout = QVBoxLayout()

#     def make_acquisition_layout(self):
#         layout = QVBoxLayout()
#         label = QLabel("Data Acquisition")
#         font = label.font()
#         font.setPointSize(12)
#         label.setFont(font)
#         filepathEntry = QLineEdit()
#         filepathEntry.setPlaceholderText()
#         filepathEntry.setPlaceholderText("acquisition filepath")

#         button = QPushButton("Open Bonsai")
#         button.clicked.connect(self.launch_bonsai_acquisition())
#         return layout
    
#     def launch_bonsai_acquisition(self):






# app = QApplication(sys.argv)

# window = MainWindow()
# window.show()

# app.exec()