# Copyright 2024
# Bryce's Senior Project
# Description: Test for the FPGA I2C and angle code
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
angle = 100

# Set brake_n = 1, enable = 1, direction = 0, angle[11:8]
control_val = ((1<<7) | (1<<6) | (0<<5) | ((angle & 0xF00) >> 8))
print("Writing control_val = " + hex(control_val))
fpga.fpgaWrite(Constants.Constants.ROTATION0_CONTROL_ADDR, control_val)

# Set angle[7:0]
target_val = (angle & 0xFF)
print("Writing target_val = " + hex(target_val))
fpga.fpgaWrite(Constants.Constants.ROTATION0_TARGET_ANGLE_ADDR, target_val)

# Confirm the data
print("ROTATION0_CONTROL_ADDR.data        = " + hex(fpga.fpgaRead(Constants.Constants.ROTATION0_CONTROL_ADDR)))
print("ROTATION0_CURRENT_ANGLE2_ADDR.data = " + hex(fpga.fpgaRead(Constants.Constants.ROTATION0_TARGET_ANGLE_ADDR)))

# Start the rotation
fpga.fpgaWrite(Constants.Constants.ROTATION0_CURRENT_ANGLE2_ADDR, 0x20)
time.sleep(1)
fpga.fpgaWrite(Constants.Constants.ROTATION0_CURRENT_ANGLE2_ADDR, 0x0)

try:
    while True:

        time.sleep(1)

        lsb_data = fpga.fpgaRead(Constants.Constants.ROTATION0_CURRENT_ANGLE_ADDR)
        msb_data = fpga.fpgaRead(Constants.Constants.ROTATION0_CURRENT_ANGLE2_ADDR)

        rd_data = ((msb_data << 8) | lsb_data) & 0xFFF

        #print('msb data = ' + hex(msb_data[1]) + ", lsb_data = " + hex(lsb_data[1]))

        print("Target angle  = " + hex(angle & 0xFFF))
        print("Current angle = " + hex(rd_data))

        # If we hit the target, then flip the target
        if(abs((angle & 0xFFF) - (rd_data & 0xFFF)) < 10 ):
            print("--- Found angle, flipping target ---")
            if(angle == 100):
                angle = 200
            else:
                angle = 100

            fpga.fpgaWrite(Constants.Constants.ROTATION0_CURRENT_ANGLE2_ADDR, 0x20)
            time.sleep(1)
            fpga.fpgaWrite(Constants.Constants.ROTATION0_CURRENT_ANGLE2_ADDR, 0x0)

        rd_data = 0
        rd_data = rd_data | (fpga.fpgaRead(Constants.Constants.DEBUG0_STATUS_ADDR) << 0)
        rd_data = rd_data | (fpga.fpgaRead(Constants.Constants.DEBUG1_STATUS_ADDR) << 8)
        rd_data = rd_data | (fpga.fpgaRead(Constants.Constants.DEBUG2_STATUS_ADDR) << 16)
        rd_data = rd_data | (fpga.fpgaRead(Constants.Constants.DEBUG3_STATUS_ADDR) << 24)
        print("Debug Data = " + hex(rd_data & 0xFFFFFFFF))

        # bits [26:24] are state
        state = (rd_data >> 16) & 0x7
        match(state):
            case 0:  print("State = IDLE")
            case 1:  print("State = CALC")
            case 2:  print("State = ACCEL")
            case 3:  print("State = CRUISE")
            case 4:  print("State = DECCEL")
            case _:  print("Invalid state!")

        

except KeyboardInterrupt:
    pass



