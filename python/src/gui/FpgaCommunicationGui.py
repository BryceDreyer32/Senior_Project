# Copyright 2024
# Bryce's Senior Project
# Description: The class which controls the PS4 
import time, sys
sys.path.append( '../constants' )
import FpgaCommunication
from Constants import Constants
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem

class FpgaCommunicationGui(FpgaCommunication.FpgaCommunication):

    # Constructor 
    def __init__(self, gui, spiChannel, spiDevice, spiMode, speed):
        self.gui = gui
        self.lines = 0

        header_labels = ["Address", "Value"]
        self.gui.regfileTable.setHorizontalHeaderLabels(header_labels)

        #self.gui.FPGA_Dialog_Box.setMaximumBlockCount(100)
        # Call superclass constructor
        super().__init__(spiChannel, spiDevice, spiMode, speed)

        self.refreshRegFileTable()


    # Override of fpgaWrite with the addition of printing to GUI console
    def fpgaWrite(self, address, data):
        # Don't print angle rotation data
        if((int(address) < Constants.ROTATION0_CONTROL_ADDR) | (int(address) > Constants.ROTATION3_PWM_TEST_ADDR)):
            self.gui.FPGA_Dialog_Box.append("Wr Addr = " + hex(address) + ", Wr Data = " + hex(data))
#        self.checkLines()
        super().fpgaWrite(address, data)

    # Override of fpgaRead with the addition of printing to GUI console
    def fpgaRead(self, address):
        val = super().fpgaRead(address)
        # Don't print angle rotation data
        if((int(address) < Constants.ROTATION0_CONTROL_ADDR) | (int(address) > Constants.ROTATION3_PWM_TEST_ADDR)):
            self.gui.FPGA_Dialog_Box.append("Rd Addr = " + hex(address) + ", Rd Data  = " + hex(val))
#        self.checkLines()
        return val

    def checkLines(self):
        self.lines +=1
        if(self.lines > 20):
            self.lines = 0
            self.gui.FPGA_Dialog_Box.pop(20)

    def refreshRegFileTable(self):
        for address in range(0, Constants.LED_CONTROL_ADDR):
            self.gui.regfileTable.setItem(address, 0, QTableWidgetItem(hex(address)))
            self.gui.regfileTable.setItem(address, 1, QTableWidgetItem(hex(self.fpgaRead(address))))