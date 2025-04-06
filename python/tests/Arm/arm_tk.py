import tkinter as tk
import sys, os, time
sys.path.append(os.path.realpath('python/src/constants'))
sys.path.append(os.path.realpath('python/src/subsystem/fpga'))
import Constants
import FpgaCommunication

# FPGA instance
fpga = FpgaCommunication.FpgaCommunication(Constants.Constants.FPGA_SPI_CHANNEL, Constants.Constants.FPGA_SPI_DEVICE, Constants.Constants.FPGA_SPI_MODE, Constants.Constants.FPGA_SPI_SPEED)
fpga.fpgaWrite(Constants.Constants.SERVO_CONTROL_ADDRESS, 0x0)

# The clock period of the servo pwm is 2^19 (the counter in the FPGA) * the period of the 27MHz FPGA clock
SERVO_PWM_PERIOD = (2**19) * (1/27e6)

# The range of servo values is 0.5 - 2.5ms over 20ms. We scale these here since we aren't exactly 20ms cycle
MIN_SERVO_COUNT = (0.5e-3 / SERVO_PWM_PERIOD) * 255
MAX_SERVO_COUNT = (2.5e-3 / SERVO_PWM_PERIOD) * 255
DELTA_SERVO_COUNT = MAX_SERVO_COUNT - MIN_SERVO_COUNT


def sliderValueToPWM(value, addr):
    match(addr):
        case Constants.Constants.BASE_SERVO_CONTROL_ADDR:
            scale1 = ((Constants.Constants.BASE_SERVO_MAX - Constants.Constants.BASE_SERVO_MIN)/100) * value + Constants.Constants.BASE_SERVO_MIN
            scale2 = ((DELTA_SERVO_COUNT/100) * scale1) + MIN_SERVO_COUNT            
            return int(scale2)
        case Constants.Constants.CENTER_SERVO_CONTROL_ADDR:
            scale1 = ((Constants.Constants.CENTER_SERVO_MAX - Constants.Constants.CENTER_SERVO_MIN)/100) * value + Constants.Constants.CENTER_SERVO_MIN
            scale2 = ((DELTA_SERVO_COUNT/100) * scale1) + MIN_SERVO_COUNT            
            return int(scale2)
        case Constants.Constants.WRIST_SERVO_CONTROL_ADDR:
            scale1 = ((Constants.Constants.WRIST_SERVO_MAX - Constants.Constants.WRIST_SERVO_MIN)/100) * value + Constants.Constants.WRIST_SERVO_MIN
            scale2 = ((DELTA_SERVO_COUNT/100) * scale1) + MIN_SERVO_COUNT            
            return int(scale2)
        case Constants.Constants.GRABBER_SERVO_CONTROL_ADDR:
            scale1 = ((Constants.Constants.GRABBER_SERVO_MAX - Constants.Constants.GRABBER_SERVO_MIN)/100) * value + Constants.Constants.GRABBER_SERVO_MIN
            scale2 = ((DELTA_SERVO_COUNT/100) * scale1) + MIN_SERVO_COUNT            
            return int(scale2)
        case _:
            return 0

def on_baseSlider_change(event):
    print("Base slider value changed to: " + str(event.widget.get()))
    value = sliderValueToPWM(event.widget.get(), Constants.Constants.BASE_SERVO_CONTROL_ADDR)
    print("Writing value " + str(value) + " to BASE_SERVO_CONTROL_ADDR")
    fpga.fpgaWrite(Constants.Constants.BASE_SERVO_CONTROL_ADDR, value)
    print("Read-back of BASE_SERVO_CONTROL_ADDR = " + str(fpga.fpgaRead(Constants.Constants.BASE_SERVO_CONTROL_ADDR)))

def on_centerSlider_change(event):
    print("Center slider value changed to: " + str(event.widget.get()))
    #fpga.fpgaWrite(Constants.Constants.CENTER_SERVO_CONTROL_ADDR, sliderValueToPWM(event.widget.get(), Constants.Constants.CENTER_SERVO_CONTROL_ADDR))

    value = sliderValueToPWM(event.widget.get(), Constants.Constants.CENTER_SERVO_CONTROL_ADDR)
    print("Writing value " + str(value) + " to CENTER_SERVO_CONTROL_ADDR")
    fpga.fpgaWrite(Constants.Constants.CENTER_SERVO_CONTROL_ADDR, value)
    print("Read-back of CENTER_SERVO_CONTROL_ADDR = " + str(fpga.fpgaRead(Constants.Constants.CENTER_SERVO_CONTROL_ADDR)))


def on_wristSlider_change(event):
    print("Wrist slider value changed to: " + str(event.widget.get()))
    fpga.fpgaWrite(Constants.Constants.WRIST_SERVO_CONTROL_ADDR, sliderValueToPWM(event.widget.get(), Constants.Constants.WRIST_SERVO_CONTROL_ADDR))

def on_grabberSlider_change(event):
    print("Grabber slider value changed to: " + str(event.widget.get()))
    fpga.fpgaWrite(Constants.Constants.GRABBER_SERVO_CONTROL_ADDR, sliderValueToPWM(event.widget.get(), Constants.Constants.GRABBER_SERVO_CONTROL_ADDR))

def enable1_click():
    current = fpga.fpgaRead(Constants.Constants.SERVO_CONTROL_ADDRESS)
    if(current & 0x1):
        fpga.fpgaWrite(Constants.Constants.SERVO_CONTROL_ADDRESS, current & 0xFE)
        print("Wrote " + str(fpga.fpgaRead(Constants.Constants.SERVO_CONTROL_ADDRESS)))
    else: 
        fpga.fpgaWrite(Constants.Constants.SERVO_CONTROL_ADDRESS, current | 0x1)
        print("Wrote " + str(fpga.fpgaRead(Constants.Constants.SERVO_CONTROL_ADDRESS)))

def enable2_click():
    current = fpga.fpgaRead(Constants.Constants.SERVO_CONTROL_ADDRESS)
    if(current & 0x4):
        fpga.fpgaWrite(Constants.Constants.SERVO_CONTROL_ADDRESS, current & 0xFB)
    else: 
        fpga.fpgaWrite(Constants.Constants.SERVO_CONTROL_ADDRESS, current | 0x4)

def enable3_click():
    current = fpga.fpgaRead(Constants.Constants.SERVO_CONTROL_ADDRESS)
    if(current & 0x2):
        fpga.fpgaWrite(Constants.Constants.SERVO_CONTROL_ADDRESS, current & 0xFD)
    else: 
        fpga.fpgaWrite(Constants.Constants.SERVO_CONTROL_ADDRESS, current | 0x2)

def enable4_click():
    current = fpga.fpgaRead(Constants.Constants.SERVO_CONTROL_ADDRESS)
    if(current & 0x8):
        fpga.fpgaWrite(Constants.Constants.SERVO_CONTROL_ADDRESS, current & 0xF7)
    else: 
        fpga.fpgaWrite(Constants.Constants.SERVO_CONTROL_ADDRESS, current | 0x8)

def on_extend_click():
    print("Extend clicked!")
    fpga.fpgaWrite(Constants.Constants.BASE_SERVO_CONTROL_ADDR, 70)
    fpga.fpgaWrite(Constants.Constants.CENTER_SERVO_CONTROL_ADDR, 25)
    fpga.fpgaWrite(Constants.Constants.WRIST_SERVO_CONTROL_ADDR, 60)
    fpga.fpgaWrite(Constants.Constants.GRABBER_SERVO_CONTROL_ADDR, 50)

def on_contract_click():
    print("Contract clicked!")
    fpga.fpgaWrite(Constants.Constants.BASE_SERVO_CONTROL_ADDR, 20)
    fpga.fpgaWrite(Constants.Constants.CENTER_SERVO_CONTROL_ADDR, 75)
    fpga.fpgaWrite(Constants.Constants.WRIST_SERVO_CONTROL_ADDR, 80)
    fpga.fpgaWrite(Constants.Constants.GRABBER_SERVO_CONTROL_ADDR, 80)

def on_disable_click():
    print("Disabling all servos, current value is " + str(fpga.fpgaRead(Constants.Constants.SERVO_CONTROL_ADDRESS)))
    fpga.fpgaWrite(Constants.Constants.SERVO_CONTROL_ADDRESS, 0x0)
    print("Servo Control register = " + str(fpga.fpgaRead(Constants.Constants.SERVO_CONTROL_ADDRESS)))

def up_nod_bite_demo_click():
    print("Go up, nod, and bite demo")
    print("Enabling all...")
    fpga.fpgaWrite(Constants.Constants.SERVO_CONTROL_ADDRESS, 0xF)

    print("Going up")
    fpga.fpgaWrite(Constants.Constants.BASE_SERVO_CONTROL_ADDR, sliderValueToPWM(20, Constants.Constants.BASE_SERVO_CONTROL_ADDR))
    fpga.fpgaWrite(Constants.Constants.CENTER_SERVO_CONTROL_ADDR, sliderValueToPWM(20, Constants.Constants.CENTER_SERVO_CONTROL_ADDR))

    time.sleep(1)

    print("Nod")
    fpga.fpgaWrite(Constants.Constants.WRIST_SERVO_CONTROL_ADDR, sliderValueToPWM(50, Constants.Constants.WRIST_SERVO_CONTROL_ADDR))
    time.sleep(0.4)
    fpga.fpgaWrite(Constants.Constants.WRIST_SERVO_CONTROL_ADDR, sliderValueToPWM(100, Constants.Constants.WRIST_SERVO_CONTROL_ADDR))
    time.sleep(0.4)
    fpga.fpgaWrite(Constants.Constants.WRIST_SERVO_CONTROL_ADDR, sliderValueToPWM(50, Constants.Constants.WRIST_SERVO_CONTROL_ADDR))
    time.sleep(0.4)
    fpga.fpgaWrite(Constants.Constants.WRIST_SERVO_CONTROL_ADDR, sliderValueToPWM(100, Constants.Constants.WRIST_SERVO_CONTROL_ADDR))

    time.sleep(0.5)

    print("Bite")
    fpga.fpgaWrite(Constants.Constants.GRABBER_SERVO_CONTROL_ADDR, sliderValueToPWM(90, Constants.Constants.GRABBER_SERVO_CONTROL_ADDR))
    time.sleep(0.4)
    fpga.fpgaWrite(Constants.Constants.GRABBER_SERVO_CONTROL_ADDR, sliderValueToPWM(0, Constants.Constants.GRABBER_SERVO_CONTROL_ADDR))
    time.sleep(0.4)
    fpga.fpgaWrite(Constants.Constants.GRABBER_SERVO_CONTROL_ADDR, sliderValueToPWM(90, Constants.Constants.GRABBER_SERVO_CONTROL_ADDR))
    time.sleep(0.4)
    fpga.fpgaWrite(Constants.Constants.GRABBER_SERVO_CONTROL_ADDR, sliderValueToPWM(0, Constants.Constants.GRABBER_SERVO_CONTROL_ADDR))
    time.sleep(0.4)

    print("Going down")
    fpga.fpgaWrite(Constants.Constants.BASE_SERVO_CONTROL_ADDR, sliderValueToPWM(0, Constants.Constants.BASE_SERVO_CONTROL_ADDR))
    fpga.fpgaWrite(Constants.Constants.CENTER_SERVO_CONTROL_ADDR, sliderValueToPWM(0, Constants.Constants.CENTER_SERVO_CONTROL_ADDR))

    time.sleep(1)

    print("Disabling all...")
    fpga.fpgaWrite(Constants.Constants.SERVO_CONTROL_ADDRESS, 0x0)


# Create the main window
root = tk.Tk()
root.title("Tkinter Grid Layout with Sliders and Buttons")

# Create the slider labels and sliders
label1 = tk.Label(root, text="Base Servo Angle")
label2 = tk.Label(root, text="Center Servo Angle")
label3 = tk.Label(root, text="Wrist Servo Angle")
label4 = tk.Label(root, text="Grabber Servo Angle")

enable1 = tk.Button(root, text="Toggle Enable", command=enable1_click)
enable2 = tk.Button(root, text="Toggle Enable", command=enable2_click)
enable3 = tk.Button(root, text="Toggle Enable", command=enable3_click)
enable4 = tk.Button(root, text="Toggle Enable", command=enable4_click)

slider1 = tk.Scale(root, from_=0, to=100, orient="horizontal")
slider2 = tk.Scale(root, from_=0, to=100, orient="horizontal")
slider3 = tk.Scale(root, from_=0, to=100, orient="horizontal")
slider4 = tk.Scale(root, from_=0, to=100, orient="horizontal")

slider1.bind("<ButtonRelease-1>", on_baseSlider_change)
slider2.bind("<ButtonRelease-1>", on_centerSlider_change)
slider3.bind("<ButtonRelease-1>", on_wristSlider_change)
slider4.bind("<ButtonRelease-1>", on_grabberSlider_change)

# Create the buttons
button1 = tk.Button(root, text="Fully Extend", command=on_extend_click)
button2 = tk.Button(root, text="Fully Contract", command=on_contract_click)
button3 = tk.Button(root, text="All disable", command=on_disable_click)

button4 = tk.Button(root, text="Nod & Bite Demo", command=up_nod_bite_demo_click)

# Place widgets in the grid
label1.grid(row=0, column=0, padx=10, pady=5, sticky="w")
enable1.grid(row=0, column=1, padx=10, pady=5)
slider1.grid(row=0, column=2, padx=10, pady=5)

label2.grid(row=1, column=0, padx=10, pady=5, sticky="w")
enable2.grid(row=1, column=1, padx=10, pady=5)
slider2.grid(row=1, column=2, padx=10, pady=5)

label3.grid(row=2, column=0, padx=10, pady=5, sticky="w")
enable3.grid(row=2, column=1, padx=10, pady=5)
slider3.grid(row=2, column=2, padx=10, pady=5)

label4.grid(row=3, column=0, padx=10, pady=5, sticky="w")
enable4.grid(row=3, column=1, padx=10, pady=5)
slider4.grid(row=3, column=2, padx=10, pady=5)

button1.grid(row=4, column=0, padx=10, pady=10, sticky="ew")
button2.grid(row=4, column=1, padx=10, pady=10, sticky="ew")
button3.grid(row=4, column=2, padx=10, pady=10, sticky="ew")

button4.grid(row=5, column=1, padx=10, pady=10, sticky="ew")

# Adjust grid weights so that the buttons expand horizontally
root.grid_rowconfigure(4, weight=1)
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)

# Start the Tkinter event loop
root.mainloop()

# Bryce's
#            slope = calculateSlope(Constants.Constants.BASE_SERVO_MIN, Constants.Constants.BASE_SERVO_MAX)
#            scale = (slope * value) + ((DELTA_SERVO_COUNT/100) * Constants.Constants.BASE_SERVO_MIN) + MIN_SERVO_COUNT
