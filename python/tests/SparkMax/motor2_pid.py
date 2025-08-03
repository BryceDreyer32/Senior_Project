# Copyright 2025
# Project Helios
# Description: This file test different motor configurations to be used for
# training the motor ANN.
# NOTE: this file is intended to be launched from the "SENIOR PROJECT"
# folder, where you can see, arduino, fpga, python folders...

# -*- coding: utf-8 -*-
import sys
import os
import time
import numpy as np
import random
import matplotlib.pyplot as plt

print(os.getcwd())
sys.path.append(os.path.realpath('python/src/constants'))
sys.path.append(os.path.realpath('python/src/subsystem/fpga'))
import Constants
import FpgaCommunication

# FPGA instance
fpga = FpgaCommunication.FpgaCommunication(Constants.Constants.FPGA_SPI_CHANNEL, Constants.Constants.FPGA_SPI_DEVICE, Constants.Constants.FPGA_SPI_MODE, Constants.Constants.FPGA_SPI_SPEED)

def run_motor(motor, speed, direction):
    start_angle = get_angle(motor)
    val = ((1 << 7) | (direction << 6) | (speed & 0x3F))
    if(motor == 0):
        fpga.fpgaWrite(Constants.Constants.ROTATION0_CONTROL_ADDR, val)

    elif(motor == 1):
        fpga.fpgaWrite(Constants.Constants.ROTATION1_CONTROL_ADDR, val)

    elif(motor == 2):
        fpga.fpgaWrite(Constants.Constants.ROTATION2_CONTROL_ADDR, val)

    elif(motor == 3):
        fpga.fpgaWrite(Constants.Constants.ROTATION3_CONTROL_ADDR, val)

def stop_motor(motor):
    if(motor == 0):
        fpga.fpgaWrite(Constants.Constants.ROTATION0_CONTROL_ADDR, 0x0)

    elif(motor == 1):
        fpga.fpgaWrite(Constants.Constants.ROTATION1_CONTROL_ADDR, 0x0)

    elif(motor == 2):
        fpga.fpgaWrite(Constants.Constants.ROTATION2_CONTROL_ADDR, 0x0)

    elif(motor == 3):
        fpga.fpgaWrite(Constants.Constants.ROTATION3_CONTROL_ADDR, 0x0)


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

def pid_controller(motor, target_angle, kp, ki, kd, previous_error, integral, dt):
    current_angle = get_angle(motor)
    error, direction = calculate_shortest_distance(current_angle, target_angle)

    integral += error * dt
    derivative = (error - previous_error) / dt
    control = kp * error + ki * integral + kd * derivative
    return direction, control, error, integral


def calculate_shortest_distance(current_angle, target_angle, encoder_resolution=4096):
    # Calculate the difference between the target and current angles
    diff = target_angle - current_angle

    # Normalize the difference to be within the range of -encoder_resolution/2 to +encoder_resolution/2
    if diff > encoder_resolution // 2:
        diff -= encoder_resolution
    elif diff < -encoder_resolution // 2:
        diff += encoder_resolution

    # Determine direction: 1 for clockwise, 0 for counterclockwise
    direction = 1 if diff > 0 else 0
    
    return abs(diff), direction

if __name__ == "__main__":
    motor = 2
    current_angle = get_angle(motor)
    
    kp = 0.02
    ki = 0.001
    kd = 0.005

    error_values = []

    # Run 11 tests with random targets
    for test in range(1,11):
        control = 0
        previous_error = 0    
        integral = 0
        dt = 20 / 1000.0
        current_dt = dt

        pv = 0
        time_steps = []
        pv_values = []
        control_values = []
        setpoint_values = []
        sleep_times = []

        target_magnitude = random.randint(200, 1200)
        target_sign = random.randint(0, 1)
        target_angle = (current_angle + target_magnitude) % 4096 if target_sign == 1 else (current_angle - target_magnitude) % 4096

        # Print the start and target
        print(f"start_angle = {current_angle}, target_angle = {target_angle}")
        # Run 50 steps to get to angle
        for steps in range(0, 50):
            start_time = time.monotonic()
            direction, control, previous_error, integral = pid_controller(motor, target_angle, kp, ki, kd, previous_error, integral, current_dt)

            # Send the control to the motor
            if(control > 8):
                control = 8
                print("Restrained")
            elif(control < 0):
                control = 0
                print("Set 0")
            run_motor(motor, int(control), direction)

            pv += control * dt  # Update process variable based on control output (simplified)
            time_steps.append(steps * dt)
            pv_values.append(get_angle(motor))
            control_values.append(control)
            setpoint_values.append(target_angle)

            elapsed_time = time.monotonic() - start_time
            sleep_time = dt - elapsed_time
            sleep_times.append(sleep_time)

            # Wait for 20ms before repeating our loop
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # If the task took longer than the interval, no sleep is needed.
                # You might want to log or handle this case if timing is critical.
                pass

            current_dt += dt

        # Force stop the motor
        stop_motor(motor)
        time.sleep(0.5)

        current_angle = get_angle(motor)
        print(f"final angle = {current_angle}, error = {target_angle-current_angle}")

        error_values.append(target_angle-current_angle)

    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 2, 1)
    plt.plot(time_steps, pv_values, label='Process Variable (PV)')
    plt.plot(time_steps, setpoint_values, label='Setpoint', linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel('Value')
    plt.title('Process Variable vs. Setpoint')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(time_steps, control_values, label='Control Output')
    plt.xlabel('Time (s)')
    plt.ylabel('Control Output')
    plt.title('Control Output over Time')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(time_steps, sleep_times, label='Sleep times')
    plt.xlabel('Steo')
    plt.ylabel('Sleep times')
    plt.title('Sleep times')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(error_values, label='Errors')
    plt.legend()

    plt.tight_layout()
    plt.show()