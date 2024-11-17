# Copyright 2024
# Bryce's Senior Project
# Description: Test for the FPGA SPI link to Orange Pi

# -*- coding: utf-8 -*-
import sys
import os
import random
sys.path.append(os.path.realpath('python/src/constants'))
sys.path.append(os.path.realpath('python/src/subsystem/fpga'))
import Constants
import FpgaCommunication

# FPGA instance
fpga = FpgaCommunication.FpgaCommunication(Constants.Constants.FPGA_SPI_CHANNEL, Constants.Constants.FPGA_SPI_DEVICE, Constants.Constants.FPGA_SPI_MODE, Constants.Constants.FPGA_SPI_SPEED)
address = 0x00

try:
    while True:
        # Write a random 16-bit value to the current address
        wr_data = random.getrandbits(16)
        fpga.fpgaWrite(address, wr_data)

        # Read back the data from the same address
        rd_data = fpga.fpgaRead(address)

        # Check if the data matches (for status registers, we expect mismatch)
        matched = (wr_data == rd_data[0])

        # Print out the result
        print("Address = " + hex(address) + ", Write = " + hex(wr_data) + ", Read = " + hex(rd_data[1]) + " " + hex(rd_data[0]) + ", Matched = " + str(matched))

        # Increment address until end of the register space, then go back to 0
        address += 1
        if(address == 0x20):
            address = 0x0

except KeyboardInterrupt:
    pass
