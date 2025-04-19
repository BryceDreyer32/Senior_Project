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

def setup():

    # Set brake_n = 0, enable = 0
    fpga.fpgaWrite(Constants.Constants.ROTATION0_CONTROL_ADDR, 0x0)

    print("--- SETTING UP PARAMETERS ---")
    enable_stall_chk = 0x0 << 1 # [1]
    fpga.fpgaWrite(Constants.Constants.ROTATION_GEN_CTRL_ADDR, enable_stall_chk)

    # Set PID coefficients
    fpga.fpgaWrite(Constants.Constants.KP_COEFFICIENT_ADDR, 0x02)
    fpga.fpgaWrite(Constants.Constants.KI_KD_COEFFICIENTS_ADDR, 0x00)


def setAngle(angle):
    print("--- SET ANGLE ---")
    print("Enabling & Writing target_val = " + hex(angle)+ " (" + str(angle) + ")")
    fpga.fpgaWrite(Constants.Constants.ROTATION0_CONTROL_ADDR, (1<<6) | ((angle & 0xF00) >> 8))
    fpga.fpgaWrite(Constants.Constants.ROTATION0_TARGET_ANGLE_ADDR, (angle & 0xFF))

    confirm = (fpga.fpgaRead(Constants.Constants.ROTATION0_CONTROL_ADDR) & 0xF) << 8
    confirm = confirm | fpga.fpgaRead(Constants.Constants.ROTATION0_TARGET_ANGLE_ADDR)
    print("Confirming target angle = " + hex(confirm) + " (" + str(confirm) + ")")

def setUpdateAngle():
    print("--- SETTING UPDATE ANGLE ---")
    # Start the rotation (by updating the target angle by writing update_angle0)
    # Just have to write a 1 to bit 5 - the conversion to a pulse is handled in hardware
    fpga.fpgaWrite(Constants.Constants.ROTATION0_CURRENT_ANGLE2_ADDR, 1<<5)

def readCurrentAngle():
    # Write bit 6 to get a snapshot into the register
    fpga.fpgaWrite(Constants.Constants.ROTATION0_CURRENT_ANGLE2_ADDR, 0x40)

    current = fpga.fpgaRead(Constants.Constants.ROTATION0_CURRENT_ANGLE_ADDR)
    current = current | (fpga.fpgaRead(Constants.Constants.ROTATION0_CURRENT_ANGLE2_ADDR) << 8)
    print("Current angle = " + str(current & 0xFFF))

def readTargetAngle():
    target = fpga.fpgaRead(Constants.Constants.ROTATION0_CONTROL_ADDR) << 8
    target = target | fpga.fpgaRead(Constants.Constants.ROTATION0_TARGET_ANGLE_ADDR)
    print("Target angle  = " + str(target & 0xFFF))

def getCurrentState():
    return fpga.fpgaRead(Constants.Constants.DEBUG2_STATUS_ADDR) & 0x3

def getPwmDirection():
    rd_data = fpga.fpgaRead(Constants.Constants.DEBUG2_STATUS_ADDR) >> 2
    print("PWM direction = " + str(rd_data & 0x1))

def getCurrStep():
    rd_data = fpga.fpgaRead(Constants.Constants.DEBUG2_STATUS_ADDR) >> 3
    print("curr_step = " + str(rd_data & 0x7))


try:
    # Initial setup and startup
    targetAngle = 500
    setup()
    setAngle(targetAngle)
    readCurrentAngle()
    setUpdateAngle()

    accelerated = False

    while True:        
        print("-----------------")
#        readTargetAngle()
        readCurrentAngle()
        getPwmDirection()
        getCurrStep()
        state = getCurrentState() 
        match(state):
            case 1:  
                print("State = ACCEL")
                accelerated = True    
            case 2:  print("State = CRUISE")
            case 3:  print("State = DECEL")
            case 0:  
                print("State = IDLE")
#                print("--- Found angle, flipping target ---")

#                fpga.fpgaWrite(Constants.Constants.ROTATION0_CONTROL_ADDR, 0x0)
                if(accelerated):
                    readCurrentAngle()

                    if(targetAngle == 200):
                        targetAngle = 3600
                    else:
                        targetAngle = 200

                    readCurrentAngle()
                    time.sleep(0.2)
                    readCurrentAngle()
                    time.sleep(0.2)
                    readCurrentAngle()
                    time.sleep(0.2)
                    readCurrentAngle()
                    time.sleep(0.2)
                    readCurrentAngle()
                    time.sleep(0.2)
                    readCurrentAngle()

                    time.sleep(3)

                    setAngle(targetAngle)
                    readCurrentAngle()
                    setUpdateAngle()
                    accelerated = False


#            case _:  print("   State             = Invalid state! " + str(state))
        
#        print("   pwm_update        = " + str((rd_data >> (16+4)) & 0x1))
#        print("   Went CALC state   = " + str((rd_data >> (16+5)) & 0x1))
#        print("   Went ACCEL state  = " + str((rd_data >> (16+6)) & 0x1))
#        print("   Went CRUISE state = " + str((rd_data >> (16+7)) & 0x1))
#        print("   pwm_direction     = " + str((rd_data >> 27) & 0x1))
#        print("   pwm_enable        = " + str((rd_data >> 29) & 0x1))
#        print("   stall_detected    = " + str((rd_data >> 30) & 0x1))
#        print(" -------------------------------------------")

        time.sleep(0.01)



except KeyboardInterrupt:
    pass

# Set brake_n = 0, enable = 0
fpga.fpgaWrite(Constants.Constants.ROTATION0_CONTROL_ADDR, 0x0)
print('Motor disabled, program exiting')


