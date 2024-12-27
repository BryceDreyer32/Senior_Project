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
import numpy as np

print(os.getcwd())
sys.path.append(os.path.realpath('python/src/constants'))
sys.path.append(os.path.realpath('python/src/subsystem/fpga'))
import Constants
import FpgaCommunication

# FPGA instance
fpga = FpgaCommunication.FpgaCommunication(Constants.Constants.FPGA_SPI_CHANNEL, Constants.Constants.FPGA_SPI_DEVICE, Constants.Constants.FPGA_SPI_MODE, Constants.Constants.FPGA_SPI_SPEED)

print("LEDs on")
fpga.fpgaWrite(Constants.Constants.LED_CONTROL_ADDR, (0x1 << 5) | (0x1 << 6))

# Set brake_n = 0, enable = 0
fpga.fpgaWrite(Constants.Constants.ROTATION0_CONTROL_ADDR, 0x0)

# Set the hammer-mode for acceleration
#enable_hammer = 0x1 << 7 # [7]
enable_hammer    = 0x0 << 7 # [7]
retry_count      = 0x2 << 5 # [6:5]
consec_chg       = 0x3 << 3 # [4:2]
enable_stall_chk = 0x0 << 1 # [1]
value = enable_hammer | retry_count | consec_chg
fpga.fpgaWrite(Constants.Constants.ROTATION_GEN_CTRL_ADDR, value)

# Set the forward and reverse steps
fwd_count = 0xF << 4 # [7:4]
rvs_count = 0x4      # [3:0]
value = fwd_count | rvs_count
fpga.fpgaWrite(Constants.Constants.HAMMER_FWD_RVS_ADDR, value)

# Set the number of times to stay at each PWM value
fpga.fpgaWrite(Constants.Constants.HAMMER_DELAY_TARGET_ADDRESS, 0x1)

# Set the offset to add to each step in the hammer & acceleration profiles
fpga.fpgaWrite(Constants.Constants.PROFILE_OFFSET_ADDR, 0)

# Set the cruise power level
fpga.fpgaWrite(Constants.Constants.CRUISE_POWER_ADDR, 25)

# Runs hot: DELAY = 0x80, OFFSET = 70, CRUISE = 100

# Set angle[7:0]
angle = 100
target_val = (angle & 0xFF)
print("Writing target_val = " + hex(target_val))
fpga.fpgaWrite(Constants.Constants.ROTATION0_TARGET_ANGLE_ADDR, target_val)

# Confirm the data
print("ROTATION0_CONTROL_ADDR.data        = " + hex(fpga.fpgaRead(Constants.Constants.ROTATION0_CONTROL_ADDR)))
print("ROTATION0_CURRENT_ANGLE2_ADDR.data = " + hex(fpga.fpgaRead(Constants.Constants.ROTATION0_TARGET_ANGLE_ADDR)))

test_profile = [[5,14,23,31,39,47,54,61,67,73,78,83,88,92,95,98],
                [5,7,9,11,14,18,25,33,45,60,72,82,90,95,98,100],
                [5,8,12,22,40,52,58,60,63,68,75,88,95,97,99,100],
                [5,7,9,11,13,15,18,21,25,30,38,48,60,80,92,100]]

np_results = np.zeros((4, 16))

for run in range(0,3):
    for test_idx in range(0,4):
        print("-------------------------------------------")
        print("STARTING TEST " + str(test_idx + 1))
        print("-------------------------------------------")

        # Program profile for test 1
        print("--- PROGRAMMING PROFILE ---")
        addr = Constants.Constants.PWM_PROFILE_BASE_ADDR
        profile = test_profile[test_idx]
        for point in profile:
            print("Writing value " + str(point) + " to address " + str(hex(addr)))
            fpga.fpgaWrite(addr, point)
            print("Read back     " + str(fpga.fpgaRead(addr)))
            addr += 1

        print("--- ENABLING ---")
        # Set brake_n = 1, enable = 1, direction = 0, angle[11:8]
        control_val = ((1<<7) | (1<<6) | (0<<5) | ((angle & 0xF00) >> 8))
        print("Writing control_val = " + hex(control_val))
        fpga.fpgaWrite(Constants.Constants.ROTATION0_CONTROL_ADDR, control_val)

        # Start the test
        fpga.fpgaWrite(Constants.Constants.ROTATION0_CURRENT_ANGLE2_ADDR, 0x20)
        fpga.fpgaWrite(Constants.Constants.ROTATION0_CURRENT_ANGLE2_ADDR, 0x0)
        time.sleep(10)

        # Set brake_n = 0, enable = 0
        print("--- DISABLING ---")
        fpga.fpgaWrite(Constants.Constants.ROTATION0_CONTROL_ADDR, 0x0)

        # Get the results
        print("--- COLLECTING RESULTS ---")
        results = []
        for addr in range(Constants.Constants.ANGLE_CHG_BASE_ADDR, Constants.Constants.ANGLE_CHG_BASE_ADDR + 8):
            result = fpga.fpgaRead(addr)
            results.append(result)
            print("Read result " + str(hex(result)) + " from address " + str(hex(addr)))

        print("--- RESULTS ---")
        total = 0
        idx = 0
        for result in results:
            low = result & 0xF
            high = (result >> 4) & 0xF
            print(str(low))
            print(str(high))
            total += low + high

            np_results[test_idx][2*idx] += low
            np_results[test_idx][2*idx+1] += high

            idx += 1

        print('Total = ' + str(total))

        print("\n\n\n")        
        time.sleep(1)
    
    time.sleep(3)

print('--- MULTI-RUN SUMMARY ---')
print(np.transpose(np_results))
print('')