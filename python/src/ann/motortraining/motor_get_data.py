# Copyright 2025
# Project Helios
# Description: This file test different motor configurations to be used for
# training the motor ANN.
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

def run_motor(motor, speed, duration):
    start_angle = get_angle(motor)
    val = ((1 << 7) | (1 << 6) | (speed & 0x3F))
    if(motor == 0):
        fpga.fpgaWrite(Constants.Constants.ROTATION0_CONTROL_ADDR, val)
        time.sleep(duration)
        fpga.fpgaWrite(Constants.Constants.ROTATION0_CONTROL_ADDR, 0x0)

    elif(motor == 1):
        fpga.fpgaWrite(Constants.Constants.ROTATION1_CONTROL_ADDR, val)
        time.sleep(duration)
        fpga.fpgaWrite(Constants.Constants.ROTATION1_CONTROL_ADDR, 0x0)

    elif(motor == 2):
        fpga.fpgaWrite(Constants.Constants.ROTATION2_CONTROL_ADDR, val)
        time.sleep(duration)
        fpga.fpgaWrite(Constants.Constants.ROTATION2_CONTROL_ADDR, 0x0)

    elif(motor == 3):
        fpga.fpgaWrite(Constants.Constants.ROTATION3_CONTROL_ADDR, val)
        time.sleep(duration)
        fpga.fpgaWrite(Constants.Constants.ROTATION3_CONTROL_ADDR, 0x0)

    time.sleep(0.8)
    end_angle = get_angle(motor)

    if(start_angle > end_angle):         
        delta = 4096 - start_angle + end_angle
    else:
        delta = end_angle - start_angle
    return start_angle, end_angle, delta

def get_angle(motor):
    if(motor == 0):
        fpga.fpgaWrite(Constants.Constants.ROTATION0_CURRENT_ANGLE2_ADDR, 0x40)
        lsb_data = fpga.fpgaRead(Constants.Constants.ROTATION0_CURRENT_ANGLE_ADDR)
        msb_data = fpga.fpgaRead(Constants.Constants.ROTATION0_CURRENT_ANGLE2_ADDR)

    elif(motor == 1):
        fpga.fpgaWrite(Constants.Constants.ROTATION1_CURRENT_ANGLE2_ADDR, 0x40)
        lsb_data = fpga.fpgaRead(Constants.Constants.ROTATION1_CURRENT_ANGLE_ADDR)
        msb_data = fpga.fpgaRead(Constants.Constants.ROTATION1_CURRENT_ANGLE2_ADDR)

    elif(motor == 2):
        fpga.fpgaWrite(Constants.Constants.ROTATION2_CURRENT_ANGLE2_ADDR, 0x40)
        lsb_data = fpga.fpgaRead(Constants.Constants.ROTATION2_CURRENT_ANGLE_ADDR)
        msb_data = fpga.fpgaRead(Constants.Constants.ROTATION2_CURRENT_ANGLE2_ADDR)

    elif(motor == 3):
        fpga.fpgaWrite(Constants.Constants.ROTATION3_CURRENT_ANGLE2_ADDR, 0x40)
        lsb_data = fpga.fpgaRead(Constants.Constants.ROTATION3_CURRENT_ANGLE_ADDR)
        msb_data = fpga.fpgaRead(Constants.Constants.ROTATION3_CURRENT_ANGLE2_ADDR)

    rd_data = ((msb_data << 8) | lsb_data) & 0xFFF
    return rd_data

def main():
    with open('python/src/ann/motortraining/motor2_training_results.txt', 'w') as f:
        #f.write("Motor Training Results:\n")
        f.write("Motor\tTrial\tSpeed\tDuration\tStart Angle\tEnd Angle\tAngle Change\n")
        motor = 2
        print(f"Testing motor {motor}...")
        for duration in np.arange(0.03, 0.04, 0.02):
            for speed in range(10, 11, 2):  # Test speeds 
                for trial in range(200):
                    print(f"Trial {trial}: Running motor {motor} at speed {speed} for {duration}")
                    start, end, angle_change = run_motor(motor, speed, duration)
                    #print(f"Motor {motor} changed angle by {angle_change} degrees.")
                    f.write(f"{motor}\t{trial}\t{speed}\t{duration}\t{start}\t{end}\t{angle_change}\n")

    print(f"Finished testing motor {motor}.\n")
    f.close()


if __name__=="__main__": 
    main()
