# Copyright 2024
# Bryce's Senior Project
# Description: Test for the FPGA SPI link to Orange Pi
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
address = 0x04

try:
    while True:
        
        # Write a random 16-bit value to the current address
        wr_data = random.getrandbits(8)
        fpga.fpgaWrite(address, wr_data)

        # Read back the data from the same address
        rd_data = fpga.fpgaRead(address)

        # Check if the data matches (for status registers, we expect mismatch)
        matched = (wr_data == rd_data)

        # Print out the result
        print("Address = " + hex(address) + ", Write = " + hex(wr_data) + ", Read = " + hex(rd_data) + ", Matched = " + str(matched))

        # Increment address until end of the register space, then go back to 0
        if(address == 0x1f):
            address = 0x30
        elif(address == 0x30):
            address = 0x31
        elif(address == 0x31):
            address = 0x00
            print('---------------------------------------------')
            time.sleep(1)
        else:
            address += 1

        '''

        print('Slower')
        wr_data = 0x0
        fpga.fpgaWrite(0x04, wr_data)
        rd_data = fpga.fpgaRead(address)
        matched = (wr_data == rd_data[1])
        print("Address = " + hex(address) + ", Write = " + hex(wr_data) + ", Read = " + hex(rd_data[1]) + " " + hex(rd_data[0]) + ", Matched = " + str(matched))

        time.sleep(2)

        print('Faster')
        wr_data = 0x0
        fpga.fpgaWrite(0x04, wr_data)
        rd_data = fpga.fpgaRead(address)
        matched = (wr_data == rd_data[1])
        print("Address = " + hex(address) + ", Write = " + hex(wr_data) + ", Read = " + hex(rd_data[1]) + " " + hex(rd_data[0]) + ", Matched = " + str(matched))

        time.sleep(2)
        '''

except KeyboardInterrupt:
    pass
