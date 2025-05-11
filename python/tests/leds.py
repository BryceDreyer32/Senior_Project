import tkinter as tk
import sys, os, time
sys.path.append(os.path.realpath('python/src/constants'))
sys.path.append(os.path.realpath('python/src/subsystem/fpga'))
import Constants
import FpgaCommunication

intensity = 128
led_ctrl_val = 0x0

# FPGA instance
fpga = FpgaCommunication.FpgaCommunication(Constants.Constants.FPGA_SPI_CHANNEL, Constants.Constants.FPGA_SPI_DEVICE, Constants.Constants.FPGA_SPI_MODE, Constants.Constants.FPGA_SPI_SPEED)


def changeFaultStatus():
    global led_ctrl_val
    if(fault_on.get()):
        led_ctrl_val = led_ctrl_val | 0x1
        fpga.fpgaWrite(Constants.Constants.LED_CONTROL_ADDR, led_ctrl_val)
    else:
        led_ctrl_val = led_ctrl_val & (~0x1)
        fpga.fpgaWrite(Constants.Constants.LED_CONTROL_ADDR, led_ctrl_val)

def changeOpiStatus():
    global led_ctrl_val
    if(opi_on.get()):
        led_ctrl_val = led_ctrl_val | 0x2
        fpga.fpgaWrite(Constants.Constants.LED_CONTROL_ADDR, led_ctrl_val)
    else:
        led_ctrl_val = led_ctrl_val & (~0x2)
        fpga.fpgaWrite(Constants.Constants.LED_CONTROL_ADDR, led_ctrl_val)

def changePs4Status():
    global led_ctrl_val
    if(ps4_on.get()):
        led_ctrl_val = led_ctrl_val | 0x4
        fpga.fpgaWrite(Constants.Constants.LED_CONTROL_ADDR, led_ctrl_val)
    else:
        led_ctrl_val = led_ctrl_val & (~0x4)
        fpga.fpgaWrite(Constants.Constants.LED_CONTROL_ADDR, led_ctrl_val)

def on_intensitySlider_change(event):
    global led_ctrl_val
    val = int(event.widget.get())
    led_ctrl_val = (led_ctrl_val & (0x0F)) | (val << 4)
    fpga.fpgaWrite(Constants.Constants.LED_CONTROL_ADDR, led_ctrl_val)


# Create the main window
root = tk.Tk()
root.title("Tkinter Grid Layout with Sliders and Buttons")

# Variables to store the state of the checkbox
fault_on = tk.IntVar()  
opi_on = tk.IntVar()  
ps4_on = tk.IntVar()  

# Create the slider labels and sliders
label1 = tk.Label(root, text="Fault LED")
label2 = tk.Label(root, text="PS4 LED")
label3 = tk.Label(root, text="Orange Pi LED")

faultCheck = tk.Checkbutton(root ,text="Fault LED",
                        command=changeFaultStatus,onvalue=1,
                        offvalue=0,variable=fault_on)

opiCheck = tk.Checkbutton(root ,text="Orange Pi LED",
                        command=changeOpiStatus,onvalue=1,
                        offvalue=0,variable=opi_on)

ps4Check = tk.Checkbutton(root ,text="PS4 LED",
                        command=changePs4Status,onvalue=1,
                        offvalue=0,variable=ps4_on)

intensitySlider = tk.Scale(root, from_=0, to=15, orient="horizontal")
intensitySlider.bind("<ButtonRelease-1>", on_intensitySlider_change)

# Place widgets in the grid
faultCheck.grid(row=0, column=0, padx=10, pady=5, sticky="w")
opiCheck.grid(row=0, column=1, padx=10, pady=5)
ps4Check.grid(row=0, column=2, padx=10, pady=5)
intensitySlider.grid(row=1, column=0, padx=10, pady=5)
   

# Start the Tkinter event loop
root.mainloop()

