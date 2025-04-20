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
numRotations = 2
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

def on_numRotations_change(event):
    global numRotations
    numRotations = event.widget.get()

def readCurrentAngle():
    # Write bit 6 to get a snapshot into the register
    fpga.fpgaWrite(Constants.Constants.ROTATION0_CURRENT_ANGLE2_ADDR, 0x40)

    current = fpga.fpgaRead(Constants.Constants.ROTATION0_CURRENT_ANGLE_ADDR)
    current = current | (fpga.fpgaRead(Constants.Constants.ROTATION0_CURRENT_ANGLE2_ADDR) << 8)
    print("Current angle = " + str(current & 0xFFF))
    return current & 0xFFF
       

def on_start_click():
#    fpga.fpgaWrite(Constants.Constants.ROTATION_PWM_TEST_ADDR, 0)
    print("Start clicked")
    print("Writing value: " + hex(powerValue))
    fpga.fpgaWrite(Constants.Constants.ROTATION_PWM_TEST_ADDR, powerValue)    

#    val = fpga.fpgaRead(Constants.Constants.ROTATION_PWM_TEST_ADDR)
#    print("Reading back value just written: " + hex(val))

    rotationsDone = 0
    logging = False
    global array
    global data

    if dir == COUNTERCLOCKWISE:
        prevValue = 0
        rotationsDone = 0
        justPassed0 = False
        # Look for the point where we cross over 0, then rotate once again, 
        # and then start logging
        # Once we reach the configured number of rotations, we exit
        while True:
            newValue = readCurrentAngle()
            if((not justPassed0) & (newValue > 3500) & (prevValue < 500)):
                print("Crossed 0")
                justPassed0 = True
                rotationsDone += 1
                # Rotate twice to get momemntum out of the equation
                if(rotationsDone == 2):
                    print("Logging started")
                    logging = True
                # Every time a rotation completes, copy the data from array into data
                # and flush the array
                elif(rotationsDone > 2):
                    data.append(array)
                    array = []
                # If we've hit the number of configured rotations, then exit
                if(rotationsDone == numRotations+2):
                    logging = False
                    break
            elif(justPassed0 & (newValue > 2500) & (prevValue > 3000)):
                justPassed0 = False

            if(logging):
                array.append(newValue)
            prevValue = newValue
            time.sleep(0.01)

    else:
        prevValue = 0
        rotationsDone = 0
        justPassed0 = False
        # Look for the point where we cross over 0, then rotate once again, 
        # and then start logging
        # Once we reach the configured number of rotations, we exit
        while True:
            newValue = readCurrentAngle()
            if((not justPassed0) & (newValue > 0) & (prevValue > 4000) & (prevValue < 4095)):
                print("Crossed 0")
                justPassed0 = True
                rotationsDone += 1
                if(rotationsDone == 2):
                    print("Logging started")
                    logging = True
                if(rotationsDone == numRotations+2):
                    logging = False
                    break
            elif(justPassed0 & (newValue > 1000) & (prevValue > 100)):
                justPassed0 = False

            if(logging):
                array.append(newValue)
            prevValue = newValue
            time.sleep(0.01)        

    fpga.fpgaWrite(Constants.Constants.ROTATION_PWM_TEST_ADDR, STOP_VALUE)

    # Write data to a file
    print("Writing to file")
    with open("logfile.txt", "w") as file:
        for item in array:
            file.write(str(item) + '\n')
    file.close()

def on_plot():
    elms = 0
    num = 0
    for array in data:
        if(len(array) > elms):
            elms = len(array)
        num += 1
    npdata = np.zeros((elms, num))
    npdata = np.array(data, dtype=np.int16)
    # Create a line plot
    plt.plot(npdata.T)
    plt.title('Rotation angle vs Time')
    plt.xlabel('Time')
    plt.ylabel('Angle')
    plt.grid(True)
    plt.show()


def on_force_stop():
    print("Force stop clicked")
    fpga.fpgaWrite(Constants.Constants.ROTATION_PWM_TEST_ADDR, STOP_VALUE)
    slider1.set(STOP_VALUE)


def setup():
    # Set the cruise power level
    fpga.fpgaWrite(Constants.Constants.ROTATION_PWM_TEST_ADDR, STOP_VALUE)

# Run the setup
setup()


# Create the main window
root = tk.Tk()
root.title("PWM Test for SparkMax motor control")

# Create the slider labels and sliders
label1 = tk.Label(root, text="Set motor power")
label2 = tk.Label(root, text="Set number of rotations")

slider1 = tk.Scale(root, from_=1, to=127, orient="horizontal", length=300)
slider1.set(STOP_VALUE)
slider1.bind("<ButtonRelease-1>", on_motorPower_change)

slider2 = tk.Scale(root, from_=1, to=20, orient="horizontal", length=100)
slider2.set(numRotations)
slider2.bind("<ButtonRelease-1>", on_numRotations_change)

button1 = tk.Button(root, text="Start Test", command=on_start_click)
button2 = tk.Button(root, text="Force Stop", command=on_force_stop)
button3 = tk.Button(root, text="Plot Data", command=on_plot)

# Place widgets in the grid
label1.grid(row=0, column=0, padx=10, pady=5, sticky="w")
slider1.grid(row=0, column=2, padx=10, pady=5)

label2.grid(row=1, column=0, padx=10, pady=5, sticky="w")
slider2.grid(row=1, column=2, padx=10, pady=5)
button1.grid(row=3, column = 0, padx=10, pady=10)
button2.grid(row=3, column = 1, padx=10, pady=10)
button3.grid(row=3, column = 2, padx=10, pady=10)

# Adjust grid weights so that the buttons expand horizontally
root.grid_rowconfigure(4, weight=1)
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)

# Start the Tkinter event loop
root.mainloop()
