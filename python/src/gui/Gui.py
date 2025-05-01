# Copyright 2024
# Bryce's Senior Project
# Description: This is the main entry point of the Robot code
from PyQt5 import QtWidgets, uic
from PyQt5 import uic
import sys, os

class Gui(QtWidgets.QDialog):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        print("cwd = " + str(os.getcwd()))

        # Load the UI Page - added path too
        ui_path = os.path.dirname(os.path.abspath(__file__))
        uic.loadUi("GUIs/PyQt5/Rev2_Onyx.ui", self)

        self.Rotation_Button_1.clicked.connect(self.clickme)
        print("--")
 
    # action method
    def clickme(self):
 
        # printing pressed
        print("pressed")

