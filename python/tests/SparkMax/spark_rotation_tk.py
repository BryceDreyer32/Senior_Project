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
#    fpga.fpgaWrite(Constants.Constants.ROTATION0_PWM_TEST_ADDR, 0)
    print("Start clicked")
    print("Writing value: " + hex(powerValue))
    fpga.fpgaWrite(Constants.Constants.ROTATION0_PWM_TEST_ADDR, powerValue)    

#    val = fpga.fpgaRead(Constants.Constants.ROTATION0_PWM_TEST_ADDR)
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

    fpga.fpgaWrite(Constants.Constants.ROTATION0_PWM_TEST_ADDR, STOP_VALUE)

    # Write data to a file
    print("Writing to file")
    with open("logfile.txt", "w") as file:
        for array in data:
            for item in array:
                file.write(str(item) + ',')
            file.write('\n')
    file.close()

def readResultFile():
    global data
    with open("10.txt", "r") as file:
        for line in file:
            array = line.split(',')
            data.append(array)
    file.close()

def fit_line_fixed_intercept(x, y, intercept):
  """
  Fits a line to the data (x, y) with a fixed intercept.

  Args:
    x: Array of x-values.
    y: Array of y-values.
    intercept: The fixed y-intercept value.

  Returns:
    The slope of the fitted line.
  """
  # Adjust y values by subtracting the intercept
  y_adjusted = y - intercept
  
  # Calculate the slope using linear regression without an intercept
  slope = np.sum(x * y_adjusted) / np.sum(x**2)
  
  return slope

def on_plot():
    TIMESTEP = 0.01
    elms = 0
    num = 0
    for array in data:
        if(len(array) > elms):
            elms = len(array)
        num += 1
    npdata = np.zeros((num, elms))
    time = np.zeros(elms)
    for x in range(0,num):
        elms = len(data[x])
        for y in range(0,elms):
            if(data[x][y] != "\n"):
                npdata[x, y] = int(data[x][y])
                time[y] = y * TIMESTEP
    
    # Create a line plot
    plt.subplot(2, 2, 1)
    plt.plot(time, npdata.T)
    plt.title('Rotation angle vs Time')
    plt.xlabel('Time')
    plt.ylabel('Angle')
    plt.grid(True)

    # Average plot, with fit line
    average = np.average(npdata, axis=0)
    plt.subplot(2, 2, 2)

#    coef = np.polyfit(time, average.T, 1)
#    poly1d_fn = np.poly1d(coef)
#    slope = fit_line_fixed_intercept(time, average.T, 4095)
    slope = -4095 / (elms * TIMESTEP)
    poly1d_fn = np.poly1d([slope, 4095])
    plt.plot(time, average.T, time, poly1d_fn(time))
    plt.title('Average Rotation angle vs Time')
    plt.xlabel('Time')
    plt.ylabel('Angle')
    plt.grid(True)

    # Calculate the offsets at 8 intervals
    boostOffset = []
    arrayIdx = 0
    prevNum = 0
    for idx in range(0, 8):
        upperVal = 4096 - (idx*512)
        lowerVal = 4096 - ((idx+1)*512)
        sum = 0
        num = 0
        while True:
            if(arrayIdx >= elms):
                break
            ave = float(average[arrayIdx])
            if((ave <= upperVal) & (ave > lowerVal)):
                sum += ave
                num += 1
                arrayIdx += 1
            else:
                break
        t = (((num/2)+prevNum)*TIMESTEP)
        #print('t = ' + str(t))
        ideal = t * slope + 4095
        prevNum += num
#        offset = int((sum/num) - ideal)
#        boostOffset.append(offset)
#        print("Adding boost offset " + str(offset))

    derivative = np.gradient(average)
    plt.subplot(2,2,3)
    plt.plot(average, derivative)
    plt.title('Derivative of Average Rotation angle vs Angle')
    plt.xlabel('Average Angle')
    plt.ylabel('Change in Angle')
    plt.grid(True)   

    min = np.min(derivative)
    derivative -= min
#    max = np.max(derivative)
#    derivative /= max
    plt.subplot(2,2,4)
    plt.plot(average, derivative)
    plt.title('Derivative of Average Rotation angle vs Angle')
    plt.xlabel('Average Angle')
    plt.ylabel('Change in Angle')
    plt.grid(True)   

    # elms = 40, to go from 4095 -> 0
    # each sample is 4095/40 = 103 angle counts
    # so to get every 512th, you would take the 512/103 = 5th sample
    stride = int(512/(4096/elms))
    ang = 0
    for sample in range(0, elms, stride):
        DIV = 8
        print("Boost offset for angle " + str(ang*512) + " should be " + str(int(derivative[sample]/DIV)))
        ang += 1

    plt.show()

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
