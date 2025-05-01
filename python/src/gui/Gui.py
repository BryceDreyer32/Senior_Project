# Copyright 2024
# Bryce's Senior Project
# Description: This is the main entry point of the Robot code
from PyQt5 import QtWidgets, uic
from PyQt5 import uic
import sys, os
sys.path.append(os.path.realpath('python/src/constants'))
sys.path.append(os.path.realpath('python/src/subsystem/fpga'))
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
        self.Rotation_Button_2.clicked.connect(self.toggleRotation2)
        self.Rotation_Button_3.clicked.connect(self.toggleRotation3)
        self.Rotation_Button_4.clicked.connect(self.toggleRotation4)

        self.Drive_Button_1.clicked.connect(self.toggleDrive1)
        self.Drive_Button_2.clicked.connect(self.toggleDrive2)
        self.Drive_Button_3.clicked.connect(self.toggleDrive3)
        self.Drive_Button_4.clicked.connect(self.toggleDrive4)

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

    def toggleRotation2(self):
        if(self.rot2):
            self.Rotation_Button_2.setText("Start Rotation")
            self.fpga.fpgaWrite(Constants.Constants.ROTATION1_CURRENT_ANGLE2_ADDR, 0x0)
        else:
            self.Rotation_Button_2.setText("Stop Rotation")
            # Power value of 5, direction = 1, override pwm = 1
            powerValue = 5 | (1 << 6) | (1 << 7)
            self.fpga.fpgaWrite(Constants.Constants.ROTATION1_CURRENT_ANGLE2_ADDR, powerValue)
        self.rot2 = not self.rot2

    def toggleRotation3(self):
        if(self.rot3):
            self.Rotation_Button_3.setText("Start Rotation")
            self.fpga.fpgaWrite(Constants.Constants.ROTATION2_CURRENT_ANGLE2_ADDR, 0x0)
        else:
            self.Rotation_Button_3.setText("Stop Rotation")
            # Power value of 5, direction = 1, override pwm = 1
            powerValue = 5 | (1 << 6) | (1 << 7)
            self.fpga.fpgaWrite(Constants.Constants.ROTATION2_CURRENT_ANGLE2_ADDR, powerValue)
        self.rot3 = not self.rot3

    def toggleRotation4(self):
        if(self.rot4):
            self.Rotation_Button_4.setText("Start Rotation")
            self.fpga.fpgaWrite(Constants.Constants.ROTATION3_CURRENT_ANGLE2_ADDR, 0x0)
        else:
            self.Rotation_Button_4.setText("Stop Rotation")
            # Power value of 5, direction = 1, override pwm = 1
            powerValue = 5 | (1 << 6) | (1 << 7)
            self.fpga.fpgaWrite(Constants.Constants.ROTATION3_CURRENT_ANGLE2_ADDR, powerValue)
        self.rot4 = not self.rot4


    def toggleDrive1(self):
        if(self.drv1):
            self.Rotation_Button_1.setText("Start Drive")
            self.fpga.fpgaWrite(Constants.Constants.DRIVE0_CONTROL_ADDR, 0x0)
        else:
            self.Rotation_Button_1.setText("Stop Drive")
            # Power value of 5, direction = 1, override pwm = 1
            powerValue = 5 | (1 << 6)
            self.fpga.fpgaWrite(Constants.Constants.DRIVE0_CONTROL_ADDR, powerValue)
        self.drv1 = not self.drv1

    def toggleDrive2(self):
        if(self.drv2):
            self.Rotation_Button_2.setText("Start Drive")
            self.fpga.fpgaWrite(Constants.Constants.DRIVE1_CONTROL_ADDR, 0x0)
        else:
            self.Rotation_Button_2.setText("Stop Drive")
            # Power value of 5, direction = 1, override pwm = 1
            powerValue = 5 | (1 << 6)
            self.fpga.fpgaWrite(Constants.Constants.DRIVE1_CONTROL_ADDR, powerValue)
        self.drv2 = not self.drv2

    def toggleDrive3(self):
        if(self.drv3):
            self.Rotation_Button_3.setText("Start Drive")
            self.fpga.fpgaWrite(Constants.Constants.DRIVE2_CONTROL_ADDR, 0x0)
        else:
            self.Rotation_Button_3.setText("Stop Drive")
            # Power value of 5, direction = 1, override pwm = 1
            powerValue = 5 | (1 << 6)
            self.fpga.fpgaWrite(Constants.Constants.DRIVE2_CONTROL_ADDR, powerValue)
        self.drv3 = not self.drv3

    def toggleDrive4(self):
        if(self.drv4):
            self.Rotation_Button_4.setText("Start Drive")
            self.fpga.fpgaWrite(Constants.Constants.DRIVE3_CONTROL_ADDR, 0x0)
        else:
            self.Rotation_Button_4.setText("Stop Drive")
            # Power value of 5, direction = 1, override pwm = 1
            powerValue = 5 | (1 << 6)
            self.fpga.fpgaWrite(Constants.Constants.DRIVE3_CONTROL_ADDR, powerValue)
        self.drv4 = not self.drv4
