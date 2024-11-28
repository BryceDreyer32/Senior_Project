# Copyright 2024
# Bryce's Senior Project
# Description: Test for the LEDs on the motherboard
# NOTE: this file is intended to be launched from the "SENIOR PROJECT"
# folder, where you can see, arduino, fpga, python folders...

# -*- coding: utf-8 -*-
import sys
import os
import random
import time

print(os.getcwd())
sys.path.append(os.path.realpath('python/src/constants'))
sys.path.append(os.path.realpath('python/src/subsystem/fpga'))
import Constants
import FpgaCommunication

# FPGA instance
fpga = FpgaCommunication.FpgaCommunication(Constants.Constants.FPGA_SPI_CHANNEL, Constants.Constants.FPGA_SPI_DEVICE, Constants.Constants.FPGA_SPI_MODE, Constants.Constants.FPGA_SPI_SPEED)
addr = 0x25

try:
    while True:
        value = 0x1F
        fpga.fpgaWrite(addr, value)
        print("Wrote value = " + str(value) + ", Read value = " + hex(fpga.fpgaRead(addr)[1]))
        time.sleep(1)

        value = 0x10
        fpga.fpgaWrite(addr, value)
        print("Wrote value = " + str(value) + ", Read value = " + hex(fpga.fpgaRead(addr)[1]))
        time.sleep(1)


except KeyboardInterrupt:
    pass
