import tkinter as tk
import sys, os, time
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.realpath('python/src/constants'))
sys.path.append(os.path.realpath('python/src/subsystem/fpga'))
import Constants
import FpgaCommunication

STOP_VALUE = 64
powerValue = 0
dir = 0
array = []
data = []
numRotations = 5
CLOCKWISE = 0
COUNTERCLOCKWISE = 1

# FPGA instance
fpga = FpgaCommunication.FpgaCommunication(Constants.Constants.FPGA_SPI_CHANNEL, Constants.Constants.FPGA_SPI_DEVICE, Constants.Constants.FPGA_SPI_MODE, Constants.Constants.FPGA_SPI_SPEED)

def on_motorPower_change(event):
    print("Power slider value changed to: " + str(event.widget.get()))
    getval = event.widget.get()
    value = getval if getval > 64 else 64 - getval
    global dir
    dir = 1 if value > 64 else 0
    global powerValue 
    powerValue = value | (dir << 6) | (1 << 7)
    print("powerValue = " + hex(powerValue))    

def on_start_click():
#    fpga.fpgaWrite(Constants.Constants.ROTATION0_PWM_TEST_ADDR, 0)
    print("Start clicked")
    print("Writing value: " + hex(powerValue))
    fpga.fpgaWrite(Constants.Constants.ROTATION0_PWM_TEST_ADDR, powerValue)    
    print("Reading back: " + hex(fpga.fpgaRead(Constants.Constants.ROTATION0_PWM_TEST_ADDR)))


def on_force_stop():
    print("Force stop clicked")
    fpga.fpgaWrite(Constants.Constants.ROTATION0_PWM_TEST_ADDR, STOP_VALUE)
    slider1.set(STOP_VALUE)


def setup():
    # Set the cruise power level
    fpga.fpgaWrite(Constants.Constants.ROTATION0_PWM_TEST_ADDR, STOP_VALUE)

#    fpga.fpgaWrite(Constants.Constants.LOWER_BOUND1_ADDR, 750>>4)
#    fpga.fpgaWrite(Constants.Constants.UPPER_BOUND1_ADDR, 800>>4)
#    fpga.fpgaWrite(Constants.Constants.LOWER_BOUND2_ADDR, 2800>>4)
#    fpga.fpgaWrite(Constants.Constants.UPPER_BOUND2_ADDR, 2900>>4)
#    fpga.fpgaWrite(Constants.Constants.BOOST_ADDR, 15)   


# Run the setup
setup()
#readResultFile()

# Create the main window
root = tk.Tk()
root.title("PWM Test for SparkMax motor control")

# Create the slider labels and sliders
label1 = tk.Label(root, text="Set motor power")
label2 = tk.Label(root, text="Set number of rotations")

slider1 = tk.Scale(root, from_=1, to=127, orient="horizontal", length=300)
slider1.set(STOP_VALUE)
slider1.bind("<ButtonRelease-1>", on_motorPower_change)

button1 = tk.Button(root, text="Start Test", command=on_start_click)
button2 = tk.Button(root, text="Force Stop", command=on_force_stop)

# Place widgets in the grid
label1.grid(row=0, column=0, padx=10, pady=5, sticky="w")
slider1.grid(row=0, column=2, padx=10, pady=5)

button1.grid(row=3, column = 0, padx=10, pady=10)
button2.grid(row=3, column = 1, padx=10, pady=10)

# Adjust grid weights so that the buttons expand horizontally
root.grid_rowconfigure(4, weight=1)
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)

# Start the Tkinter event loop
root.mainloop()
