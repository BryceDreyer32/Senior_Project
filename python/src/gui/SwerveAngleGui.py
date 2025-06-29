# Copyright 2024
# Bryce's Senior Project
# Description: The class which controls the PS4 
import time, sys
sys.path.append( '../constants' )
import FpgaCommunication
from Constants import Constants
from PyQt5.QtCore import QThread, pyqtSignal

class SwerveAngleGui(QThread):
    angle_data = pyqtSignal(str)

    def __init__(self, fpga):            
        super().__init__()
        self.fpga = fpga

    def run(self):
        while(1):
            data = ""

            # Write to bit 6 to initiate a capture, then read back [7:0] followed by [11:8]
            # Set bit[7] to enable the i2c 
            self.fpga.fpgaWrite(Constants.ROTATION0_CONTROL_ADDR, 0xC0) 
            self.fpga.fpgaWrite(Constants.ROTATION0_CURRENT_ANGLE2_ADDR, 0x40) 
            val = self.fpga.fpgaRead(Constants.ROTATION0_CURRENT_ANGLE_ADDR) 
            val |= (self.fpga.fpgaRead(Constants.ROTATION0_CURRENT_ANGLE2_ADDR) & 0x0F) << 8
            data += str(val) + ","

            self.fpga.fpgaWrite(Constants.ROTATION1_CONTROL_ADDR, 0xC0) 
            self.fpga.fpgaWrite(Constants.ROTATION1_CURRENT_ANGLE2_ADDR, 0x40) 
            val = self.fpga.fpgaRead(Constants.ROTATION1_CURRENT_ANGLE_ADDR) 
            val |= (self.fpga.fpgaRead(Constants.ROTATION1_CURRENT_ANGLE2_ADDR) & 0x0F) << 8
            data += str(val) + ","

            self.fpga.fpgaWrite(Constants.ROTATION2_CONTROL_ADDR, 0xC0) 
            self.fpga.fpgaWrite(Constants.ROTATION2_CURRENT_ANGLE2_ADDR, 0x40) 
            val = self.fpga.fpgaRead(Constants.ROTATION2_CURRENT_ANGLE_ADDR) 
            val |= (self.fpga.fpgaRead(Constants.ROTATION2_CURRENT_ANGLE2_ADDR) & 0x0F) << 8
            data += str(val) + ","

            self.fpga.fpgaWrite(Constants.ROTATION3_CONTROL_ADDR, 0xC0) 
            self.fpga.fpgaWrite(Constants.ROTATION3_CURRENT_ANGLE2_ADDR, 0x40) 
            val = self.fpga.fpgaRead(Constants.ROTATION3_CURRENT_ANGLE_ADDR) 
            val |= (self.fpga.fpgaRead(Constants.ROTATION3_CURRENT_ANGLE2_ADDR) & 0x0F) << 8
            data += str(val)

            self.angle_data.emit(data)

            time.sleep(0.2)
