import tkinter as tk
import sys, os, time
import numpy as np
sys.path.append(os.path.realpath('python/src/constants'))
sys.path.append(os.path.realpath('python/src/subsystem/fpga'))
import Constants
import FpgaCommunication

STOP_VALUE = 150

# FPGA instance
fpga = FpgaCommunication.FpgaCommunication(Constants.Constants.FPGA_SPI_CHANNEL, Constants.Constants.FPGA_SPI_DEVICE, Constants.Constants.FPGA_SPI_MODE, Constants.Constants.FPGA_SPI_SPEED)

def on_motorPower_change(event):
    print("Power slider value changed to: " + str(event.widget.get()))
    fpga.fpgaWrite(Constants.Constants.CRUISE_POWER_ADDR, event.widget.get())

    val = fpga.fpgaRead(Constants.Constants.CRUISE_POWER_ADDR)
    print("Reading back value just written: " + str(val))    

def on_shutdown_click():
    fpga.fpgaWrite(Constants.Constants.ROTATION0_CONTROL_ADDR, 0)
    print("Shutdown clicked")

def on_re_enable_click():
    print("Re-enable clicked")
    setup()
    fpga.fpgaWrite(Constants.Constants.CRUISE_POWER_ADDR, STOP_VALUE)
    slider1.set(STOP_VALUE)


def setup():
    print("--- PROGRAMMING PROFILE ---")
    addr = Constants.Constants.PWM_PROFILE_BASE_ADDR
    for point in range(16):
        fpga.fpgaWrite(addr, STOP_VALUE)
        addr += 1
    # Disable
    fpga.fpgaWrite(Constants.Constants.ROTATION0_CONTROL_ADDR, 0x0)

    print("--- SETTING UP PARAMETERS ---")
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
    fpga.fpgaWrite(Constants.Constants.HAMMER_DELAY_TARGET_ADDRESS, 0x01)

    # Set the offset to add to each step in the hammer & acceleration profiles
    fpga.fpgaWrite(Constants.Constants.PROFILE_OFFSET_ADDR, 0)

    # Set the cruise power level
    fpga.fpgaWrite(Constants.Constants.CRUISE_POWER_ADDR, STOP_VALUE)

    # Need angle set, otherwise nothing runs!
    # Set angle[7:0]
    angle = 100
    target_val = (angle & 0xFF)
    print("Writing target_val = " + hex(target_val))
    fpga.fpgaWrite(Constants.Constants.ROTATION0_TARGET_ANGLE_ADDR, target_val)

    # Confirm the data
    print("ROTATION0_CONTROL_ADDR.data        = " + hex(fpga.fpgaRead(Constants.Constants.ROTATION0_CONTROL_ADDR)))
    print("ROTATION0_CURRENT_ANGLE2_ADDR.data = " + hex(fpga.fpgaRead(Constants.Constants.ROTATION0_TARGET_ANGLE_ADDR)))


    print("--- ENABLING ---")
    # Set brake_n = 1, enable = 1, direction = 0, angle[11:8]
    control_val = ((1<<7) | (1<<6) | (0<<5) | ((0 & 0xF00) >> 8))
    print("Writing control_val = " + hex(control_val))
    fpga.fpgaWrite(Constants.Constants.ROTATION0_CONTROL_ADDR, control_val)

    print("--- RUNNING ---")
    # Start the rotation
    fpga.fpgaWrite(Constants.Constants.ROTATION0_CURRENT_ANGLE2_ADDR, 0x20)
    fpga.fpgaWrite(Constants.Constants.ROTATION0_CURRENT_ANGLE2_ADDR, 0x0)

# Run the setup
setup()

# Create the main window
root = tk.Tk()
root.title("PWM Test for SparkMax motor control")

# Create the slider labels and sliders
label1 = tk.Label(root, text="Set motor power")

slider1 = tk.Scale(root, from_=0, to=255, orient="horizontal", length=300)
slider1.set(STOP_VALUE)
slider1.bind("<ButtonRelease-1>", on_motorPower_change)

button1 = tk.Button(root, text="Shutdown", command=on_shutdown_click)
button2 = tk.Button(root, text="Re-enable", command=on_re_enable_click)

# Place widgets in the grid
label1.grid(row=0, column=0, padx=10, pady=5, sticky="w")
slider1.grid(row=0, column=2, padx=10, pady=5)
button1.grid(row=1, column = 1, padx=10, pady=10)
button2.grid(row=1, column = 2, padx=10, pady=10)

# Adjust grid weights so that the buttons expand horizontally
root.grid_rowconfigure(4, weight=1)
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)

# Start the Tkinter event loop
root.mainloop()
