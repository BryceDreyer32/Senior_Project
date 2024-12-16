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

# Set brake_n = 0, enable = 0
fpga.fpgaWrite(Constants.Constants.ROTATION0_CONTROL_ADDR, 0x0)

# Set the hammer-mode for acceleration
enable_hammer = 0x1 << 7 # [7]
retry_count   = 0x2 << 5 # [6:5]
consec_chg    = 0x3 << 3 # [5:3]
value = enable_hammer | retry_count | consec_chg
fpga.fpgaWrite(Constants.Constants.ROTATION_GEN_CTRL1_ADDR, value)

# Set the forward and reverse steps
fwd_count = 6 << 0xF # [7:4]
rvs_count = 0x4      # [3:0]
value = fwd_count | rvs_count
fpga.fpgaWrite(Constants.Constants.ROTATION_GEN_CTRL2_ADDR, value)

# Set angle[7:0]
angle = 100
target_val = (angle & 0xFF)
print("Writing target_val = " + hex(target_val))
fpga.fpgaWrite(Constants.Constants.ROTATION0_TARGET_ANGLE_ADDR, target_val)

# Confirm the data
print("ROTATION0_CONTROL_ADDR.data        = " + hex(fpga.fpgaRead(Constants.Constants.ROTATION0_CONTROL_ADDR)))
print("ROTATION0_CURRENT_ANGLE2_ADDR.data = " + hex(fpga.fpgaRead(Constants.Constants.ROTATION0_TARGET_ANGLE_ADDR)))

# Set brake_n = 1, enable = 1, direction = 0, angle[11:8]
control_val = ((1<<7) | (1<<6) | (0<<5) | ((angle & 0xF00) >> 8))
print("Writing control_val = " + hex(control_val))
fpga.fpgaWrite(Constants.Constants.ROTATION0_CONTROL_ADDR, control_val)

# Start the rotation
fpga.fpgaWrite(Constants.Constants.ROTATION0_CURRENT_ANGLE2_ADDR, 0x20)
time.sleep(1)
fpga.fpgaWrite(Constants.Constants.ROTATION0_CURRENT_ANGLE2_ADDR, 0x0)

try:
    while True:

        time.sleep(1)

        target = fpga.fpgaRead(Constants.Constants.ROTATION0_CONTROL_ADDR) << 8
        target = target | fpga.fpgaRead(Constants.Constants.ROTATION0_TARGET_ANGLE_ADDR)
        print("Target angle  = " + str(target & 0xFFF))

        current = fpga.fpgaRead(Constants.Constants.ROTATION0_CURRENT_ANGLE_ADDR)
        current = current | fpga.fpgaRead(Constants.Constants.ROTATION0_CURRENT_ANGLE2_ADDR) << 8
        print("Current angle = " + str(current & 0xFFF))

        # If we hit the target, then flip the target
        if(abs((angle & 0xFFF) - (current & 0xFFF)) < 10 ):
            print("--- Found angle, flipping target ---")
            if(angle == 100):
                angle = 800
            else:
                angle = 100

            # Brake, disable
            fpga.fpgaWrite(Constants.Constants.ROTATION0_TARGET_ANGLE_ADDR, angle & 0xFF)
            control_val = ((0<<7) | (0<<6) | (0<<5) | ((angle & 0xF00) >> 8))

            time.sleep(1)

            # UnBrake, enable
            fpga.fpgaWrite(Constants.Constants.ROTATION0_CONTROL_ADDR, control_val)
            control_val = ((1<<7) | (1<<6) | (0<<5) | ((angle & 0xF00) >> 8))
            fpga.fpgaWrite(Constants.Constants.ROTATION0_CONTROL_ADDR, control_val)

            # Write update angle
            fpga.fpgaWrite(Constants.Constants.ROTATION0_CURRENT_ANGLE2_ADDR, 0x20)


        rd_data = 0
        rd_data = rd_data | (fpga.fpgaRead(Constants.Constants.DEBUG0_STATUS_ADDR) << 0)
        rd_data = rd_data | (fpga.fpgaRead(Constants.Constants.DEBUG1_STATUS_ADDR) << 8)
        rd_data = rd_data | (fpga.fpgaRead(Constants.Constants.DEBUG2_STATUS_ADDR) << 16)
        rd_data = rd_data | (fpga.fpgaRead(Constants.Constants.DEBUG3_STATUS_ADDR) << 24)
        print("Debug Data = " + hex(rd_data & 0xFFFFFFFF))

        # bits [19:16] are state
        state = (rd_data >> 15) & 0xF
        match(state):
            case 0:  print("   State             = IDLE           ")
            case 1:  print("   State             = CALC           ")
            case 2:  print("   State             = ACCEL          ")
            case 3:  print("   State             = HAMMER_FORWARD ")
            case 4:  print("   State             = HAMMER_REVERSE ")
            case 5:  print("   State             = HAMMER_PASS    ")
            case 6:  print("   State             = HAMMER_FAIL    ")
            case 7:  print("   State             = CRUISE         ")
            case 8:  print("   State             = DECEL          ")
            case 9:  print("   State             = SHUTDOWN       ")
            case _:  print("   State             = Invalid state! ")
        
        print("   pwm_update        = " + str((rd_data >> 19) & 0x1))
        print("   chg_cnt[2:0]      = " + str((rd_data >> 20) & 0x7))
        print("   pwm_done          = " + str((rd_data >> 23) & 0x1))
        print("   abort_angle       = " + str((rd_data >> 24) & 0x1))
        print("   angle_update      = " + str((rd_data >> 25) & 0x1))
        print("   pwm_direction     = " + str((rd_data >> 26) & 0x1))
        print("   pwm_done          = " + str((rd_data >> 27) & 0x1))
        print("   retry_cnt[1:0]    = " + str((rd_data >> 28) & 0x1))
        print("   run_stall         = " + str((rd_data >> 30) & 0x1))
        print("   startup_fail      = " + str((rd_data >> 31) & 0x1))
        print(" -------------------------------------------")

        # If we're in the HAMMER_FAIL state, wait a bit and then try again
        if(state == 6):
            # Brake, disable
            fpga.fpgaWrite(Constants.Constants.ROTATION0_TARGET_ANGLE_ADDR, angle & 0xFF)
            control_val = ((0<<7) | (0<<6) | (0<<5) | ((angle & 0xF00) >> 8))

            print('HAMMER_FAIL. Retrying...')
            time.sleep(2)

            # UnBrake, enable
            fpga.fpgaWrite(Constants.Constants.ROTATION0_CONTROL_ADDR, control_val)
            control_val = ((1<<7) | (1<<6) | (0<<5) | ((angle & 0xF00) >> 8))
            fpga.fpgaWrite(Constants.Constants.ROTATION0_CONTROL_ADDR, control_val)

            # Write update angle
            fpga.fpgaWrite(Constants.Constants.ROTATION0_CURRENT_ANGLE2_ADDR, 0x0)
            fpga.fpgaWrite(Constants.Constants.ROTATION0_CURRENT_ANGLE2_ADDR, 0x20)
            fpga.fpgaWrite(Constants.Constants.ROTATION0_CURRENT_ANGLE2_ADDR, 0x0)

#assign debug_signals = {startup_fail, run_stall, retry_cnt[1:0], pwm_direction, angle_update, abort_angle, pwm_done,
#                        chg_cnt[2:0], pwm_update, ps[3:0]};

except KeyboardInterrupt:
    pass

# Set brake_n = 0, enable = 0
fpga.fpgaWrite(Constants.Constants.ROTATION0_CONTROL_ADDR, 0x0)
print('Motor disabled, program exiting')

