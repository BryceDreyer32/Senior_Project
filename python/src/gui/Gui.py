# Copyright 2024
# Bryce's Senior Project
# Description: This is the main entry point of the Robot code
from PyQt5 import QtWidgets, uic
from PyQt5 import uic
import sys, os
import Constants
import FpgaCommunication

class Gui(QtWidgets.QDialog):
    rot1 = False
    rot2 = False
    rot3 = False
    rot4 = False
    drv1 = False
    drv2 = False
    drv3 = False
    drv4 = False   

    def __init__(self, fpga):
        super().__init__()
        self.fpga = fpga

        print("cwd = " + str(os.getcwd()))

        # Load the UI Page - added path too
        ui_path = os.path.dirname(os.path.abspath(__file__))
        uic.loadUi("GUIs/PyQt5/Rev2_Onyx.ui", self)

        self.Rotation_Button_1.clicked.connect(self.toggleRotation1)
#        self.Rotation_Button_2.clicked.connect(self.toggleRotation2)
#        self.Rotation_Button_3.clicked.connect(self.toggleRotation3)
#        self.Rotation_Button_4.clicked.connect(self.toggleRotation4)
#
#        self.Drive_Button_1.clicked.connect(self.toggleDrive1)
#        self.Drive_Button_2.clicked.connect(self.toggleDrive2)
#        self.Drive_Button_3.clicked.connect(self.toggleDrive3)
#        self.Drive_Button_4.clicked.connect(self.toggleDrive4)

    def toggleRotation1(self):
        if(self.rot1):
            self.Rotation_Button_1.setText("Start Rotation")
            self.fpga.fpgaWrite(Constants.Constants.ROTATION0_CURRENT_ANGLE2_ADDR, 0x0)
        else:
            self.Rotation_Button_1.setText("Stop Rotation")
            # Power value of 5, direction = 1, override pwm = 1
            powerValue = 5 | (1 << 6) | (1 << 7)
            self.fpga.fpgaWrite(Constants.Constants.ROTATION0_CURRENT_ANGLE2_ADDR, powerValue)
        self.rot1 = not self.rot1


