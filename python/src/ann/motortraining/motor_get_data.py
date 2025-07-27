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

    if(speed < 5):
        time.sleep(0.3)
    elif(speed < 8):
        time.sleep(0.5)
    else:
        time.sleep(1)

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

def data_clean(angle_change_data):
    # Convert list to NumPy array
    angle_change_data = np.array(angle_change_data)
    
    # Filter outliers using IQR method
    Q1 = np.quantile(angle_change_data,0.25)
    Q3 = np.quantile(angle_change_data,0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Boolean mask for filtered data
    filtered_mask = (angle_change_data >= lower_bound) & (angle_change_data <= upper_bound)

    # Filtered data
    return angle_change_data[filtered_mask]

def main():
    motor = 2
    raw_lines = []
    angle_data = []
    voltage = 12.74
    print(f"Testing motor {motor}...")
    for duration in np.arange(0.04, 0.09, 0.01):
        for speed in range(3, 8, 1):  # Test speeds 
            for trial in range(50):
                print(f"Trial {trial}: Running motor {motor} at speed {speed} for {duration:.2f}")
                start, end, angle_change = run_motor(motor, speed, duration)                                    
                #f.write(f"{motor}\t{trial}\t{speed}\t{duration}\t{start}\t{end}\t{angle_change}\n")
                raw_lines.append(f"{motor},{trial},{speed},{duration:.2f},{start},{end},")
                angle_data.append(angle_change)

            # Clean the data for this setup, and write to the results file
            angle_change = data_clean(angle_data)
            with open('python/src/ann/motortraining/results2.txt', 'a') as f:
                for idx in range(0,len(angle_change)):
                    f.write(f"{raw_lines[idx]},{angle_change[idx]},{voltage}\n")
            f.close()
            raw_lines = []
            angle_data = []

    print(f"Finished testing motor {motor}.\n")




if __name__=="__main__": 
    main()
