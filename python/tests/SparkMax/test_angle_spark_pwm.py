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
#while 1:
#    print("LEDs on")
#    fpga.fpgaWrite(Constants.Constants.LED_TEST_ADDR, (0x1 << 5) | (0x1 << 6))
#    time.sleep(1)

#    print("LEDs off")
#    fpga.fpgaWrite(Constants.Constants.LED_TEST_ADDR, (0x0 << 5) | (0x0 << 6))
#    time.sleep(1)

#test_profile = [[5,14,23,31,39,47,54,61,67,73,78,83,88,92,95,98],
test_profile = [[0,3,6,9,12,15,18,21,24,27,30,33,36,39,42,45],
                [5,7,9,11,14,18,25,33,45,60,72,82,90,95,98,100],
                [5,8,12,22,40,52,58,60,63,68,75,88,95,97,99,100],
                [5,7,9,11,13,15,18,21,25,30,38,48,60,80,92,100],
                [5,7,9,12,17,30,50,80,85,80,60,55,60,90,95,100],
                [5,8,12,22,50,30,60,40,70,50,80,60,90,70,95,100]]
np_results = np.zeros((6, 16))

print("--- PROGRAMMING PROFILE ---")
addr = Constants.Constants.PWM_PROFILE_BASE_ADDR
profile = test_profile[0]
for point in profile:
    print("Writing value " + str(point) + " to address " + str(hex(addr)))
    fpga.fpgaWrite(addr, point)
    print("Read back     " + str(fpga.fpgaRead(addr)))
    addr += 1

# Set brake_n = 0, enable = 0
fpga.fpgaWrite(Constants.Constants.ROTATION0_CONTROL_ADDR, 0x0)
#print('Off')
#time.sleep(5)

print("--- SETTING UP PARAMETERS ---")
# Set the hammer-mode for acceleration
#enable_hammer = 0x1 << 7 # [7]
#enable_hammer    = 0x0 << 7 # [7]
#retry_count      = 0x2 << 5 # [6:5]
#consec_chg       = 0x3 << 3 # [4:2]
#enable_stall_chk = 0x0 << 1 # [1]
#value = enable_hammer | retry_count | consec_chg
fpga.fpgaWrite(Constants.Constants.ROTATION_GEN_CTRL_ADDR, 0)

# Set the forward and reverse steps
#fwd_count = 0xF << 4 # [7:4]
#rvs_count = 0x4      # [3:0]
#value = fwd_count | rvs_count
#fpga.fpgaWrite(Constants.Constants.HAMMER_FWD_RVS_ADDR, value)

# Set the number of times to stay at each PWM value
fpga.fpgaWrite(Constants.Constants.PROFILE_DELAY_TARGET_ADDR, 0x11)

# No offset on profile values
fpga.fpgaWrite(Constants.Constants.PROFILE_OFFSET_ADDR, 0x0)

# Set the cruise power level
fpga.fpgaWrite(Constants.Constants.CRUISE_POWER_ADDR, 30)

# Runs hot: DELAY = 0x80, OFFSET = 70, CRUISE = 100

# Set angle[7:0]
angle = 100
target_val = (angle & 0xFF)
print("Writing target_val = " + hex(target_val))
fpga.fpgaWrite(Constants.Constants.ROTATION0_TARGET_ANGLE_ADDR, target_val)
print("ROTATION0_CURRENT_ANGLE2_ADDR.data = " + hex(fpga.fpgaRead(Constants.Constants.ROTATION0_TARGET_ANGLE_ADDR)))

print("--- ENABLING ---")
# Set brake_n = 1, enable = 1, direction = 0, angle[11:8]
control_val = ((1<<7) | (1<<6) | (0<<5) | ((angle & 0xF00) >> 8))
print("Writing control_val = " + hex(control_val))
fpga.fpgaWrite(Constants.Constants.ROTATION0_CONTROL_ADDR, control_val)
print("ROTATION0_CONTROL_ADDR.data        = " + hex(fpga.fpgaRead(Constants.Constants.ROTATION0_CONTROL_ADDR)))

print("--- RUNNING ---")
# Start the rotation (by updating the target angle by writing update_angle0)
fpga.fpgaWrite(Constants.Constants.ROTATION0_CURRENT_ANGLE2_ADDR, 0x20)
#fpga.fpgaWrite(Constants.Constants.ROTATION0_CURRENT_ANGLE2_ADDR, 0x0)

time.sleep(5)

try:
    while True:

        target = fpga.fpgaRead(Constants.Constants.ROTATION0_CONTROL_ADDR) << 8
        target = target | fpga.fpgaRead(Constants.Constants.ROTATION0_TARGET_ANGLE_ADDR)
        print("Target angle  = " + str(target & 0xFFF))

        # Write bit 6 to get a snapshot into the register
        fpga.fpgaWrite(Constants.Constants.ROTATION0_CURRENT_ANGLE2_ADDR, 0x40)

        current = fpga.fpgaRead(Constants.Constants.ROTATION0_CURRENT_ANGLE_ADDR)
        current = current | (fpga.fpgaRead(Constants.Constants.ROTATION0_CURRENT_ANGLE2_ADDR) << 8)
        print("Current angle = " + str(current & 0xFFF))

        # If we hit the target, then flip the target
#        if(abs((angle & 0xFFF) - (current & 0xFFF)) < 20 ):
#            print("--- Found angle, flipping target ---")
#            if(angle == 100):
#                angle = 800
#            else:
#                angle = 100
#
#            # Brake, disable
#            fpga.fpgaWrite(Constants.Constants.ROTATION0_TARGET_ANGLE_ADDR, angle & 0xFF)
#            control_val = ((0<<7) | (0<<6) | (0<<5) | ((angle & 0xF00) >> 8))
#            fpga.fpgaWrite(Constants.Constants.ROTATION0_CONTROL_ADDR, control_val)
#
#            time.sleep(3)
#
#            # UnBrake, enable
#            control_val = ((1<<7) | (1<<6) | (0<<5) | ((angle & 0xF00) >> 8))
#            fpga.fpgaWrite(Constants.Constants.ROTATION0_CONTROL_ADDR, control_val)
#
#            # Write update angle
#            fpga.fpgaWrite(Constants.Constants.ROTATION0_CURRENT_ANGLE2_ADDR, 0x20)
#            fpga.fpgaWrite(Constants.Constants.ROTATION0_CURRENT_ANGLE2_ADDR, 0x0)


        rd_data = 0
        rd_data = rd_data | (fpga.fpgaRead(Constants.Constants.DEBUG0_STATUS_ADDR) << 0)
        rd_data = rd_data | (fpga.fpgaRead(Constants.Constants.DEBUG1_STATUS_ADDR) << 8)
        rd_data = rd_data | (fpga.fpgaRead(Constants.Constants.DEBUG2_STATUS_ADDR) << 16)
        rd_data = rd_data | (fpga.fpgaRead(Constants.Constants.DEBUG3_STATUS_ADDR) << 24)
        print("Debug Data = " + hex(rd_data & 0xFFFFFFFF))

        # bits [19:16] are state
        state = (rd_data >> 16) & 0x7
        match(state):
#            case 0:  print("   State             = IDLE           ")
            case 1:  print("   State             = CALC           ")
            case 2:  print("   State             = ACCEL          ")
            case 3:  print("   State             = CRUISE         ")
            case 4:  print("   State             = DECEL          ")
            case 0 | 5:  
                print("   State             = IDLE/SHUTDOWN       ")
                print("--- Found angle, flipping target ---")
                if(angle == 100):
                    angle = 800
                else:
                    angle = 100

                time.sleep(3)

                # Write update angle
                fpga.fpgaWrite(Constants.Constants.ROTATION0_CURRENT_ANGLE2_ADDR, 0x20)

            case _:  print("   State             = Invalid state! " + str(state))
        
        print("   pwm_update        = " + str((rd_data >> (16+4)) & 0x1))
        print("   Went CALC state   = " + str((rd_data >> (16+5)) & 0x1))
        print("   Went ACCEL state  = " + str((rd_data >> (16+6)) & 0x1))
        print("   Went CRUISE state = " + str((rd_data >> (16+7)) & 0x1))
        print("   pwm_direction     = " + str((rd_data >> 27) & 0x1))
        print("   pwm_enable        = " + str((rd_data >> 29) & 0x1))
        print("   stall_detected    = " + str((rd_data >> 30) & 0x1))
        print(" -------------------------------------------")

        time.sleep(0.05)



except KeyboardInterrupt:
    pass

# Set brake_n = 0, enable = 0
fpga.fpgaWrite(Constants.Constants.ROTATION0_CONTROL_ADDR, 0x0)
print('Motor disabled, program exiting')


