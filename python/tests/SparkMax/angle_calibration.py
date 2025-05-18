import tkinter as tk
import sys, os, time
sys.path.append(os.path.realpath('python/src/constants'))
sys.path.append(os.path.realpath('python/src/subsystem/fpga'))
import Constants
import FpgaCommunication

# FPGA instance
fpga = FpgaCommunication.FpgaCommunication(Constants.Constants.FPGA_SPI_CHANNEL, Constants.Constants.FPGA_SPI_DEVICE, Constants.Constants.FPGA_SPI_MODE, Constants.Constants.FPGA_SPI_SPEED)

def update_label_text():        
    # Write to bit 6 to initiate a capture, then read back [7:0] followed by [11:8]
    # Set bit[7] to enable the i2c 
    fpga.fpgaWrite(Constants.Constants.ROTATION0_CONTROL_ADDR, 0xC0) 
    fpga.fpgaWrite(Constants.Constants.ROTATION0_CURRENT_ANGLE2_ADDR, 0x40) 
    val = fpga.fpgaRead(Constants.Constants.ROTATION0_CURRENT_ANGLE_ADDR) 
    val |= (fpga.fpgaRead(Constants.Constants.ROTATION0_CURRENT_ANGLE2_ADDR) & 0x0F) << 8
    wheelValue1.config(text= str(val))

    fpga.fpgaWrite(Constants.Constants.ROTATION1_CONTROL_ADDR, 0xC0) 
    fpga.fpgaWrite(Constants.Constants.ROTATION1_CURRENT_ANGLE2_ADDR, 0x40) 
    val = fpga.fpgaRead(Constants.Constants.ROTATION1_CURRENT_ANGLE_ADDR) 
    val |= (fpga.fpgaRead(Constants.Constants.ROTATION1_CURRENT_ANGLE2_ADDR) & 0x0F) << 8
    wheelValue2.config(text= str(val))

    fpga.fpgaWrite(Constants.Constants.ROTATION2_CONTROL_ADDR, 0xC0) 
    fpga.fpgaWrite(Constants.Constants.ROTATION2_CURRENT_ANGLE2_ADDR, 0x40) 
    val = fpga.fpgaRead(Constants.Constants.ROTATION2_CURRENT_ANGLE_ADDR) 
    val |= (fpga.fpgaRead(Constants.Constants.ROTATION2_CURRENT_ANGLE2_ADDR) & 0x0F) << 8
    wheelValue3.config(text= str(val))

    fpga.fpgaWrite(Constants.Constants.ROTATION3_CONTROL_ADDR, 0xC0) 
    fpga.fpgaWrite(Constants.Constants.ROTATION3_CURRENT_ANGLE2_ADDR, 0x40) 
    val = fpga.fpgaRead(Constants.Constants.ROTATION3_CURRENT_ANGLE_ADDR) 
    val |= (fpga.fpgaRead(Constants.Constants.ROTATION3_CURRENT_ANGLE2_ADDR) & 0x0F) << 8
    wheelValue4.config(text= str(val))

    root.after(200, update_label_text)  # Schedule the function to run again 200ms


# Create the main window
root = tk.Tk()
root.title("Tkinter Grid Layout with Sliders and Buttons")

# Create the slider labels and sliders
wheelLabel1 = tk.Label(root, text="Top Left Wheel:     ")
wheelLabel2 = tk.Label(root, text="Top Right Wheel:    ")
wheelLabel3 = tk.Label(root, text="Bottom Left Wheel:  ")
wheelLabel4 = tk.Label(root, text="Bottom Right Wheel: ")

wheelValue1 = tk.Label(root, text="")
wheelValue2 = tk.Label(root, text="")
wheelValue3 = tk.Label(root, text="")
wheelValue4 = tk.Label(root, text="")

# Place widgets in the grid
wheelLabel1.grid(row=0, column=0, padx=10, pady=5, sticky="w")
wheelValue1.grid(row=0, column=1, padx=10, pady=5)
wheelLabel2.grid(row=1, column=0, padx=10, pady=5, sticky="w")
wheelValue2.grid(row=1, column=1, padx=10, pady=5)
wheelLabel3.grid(row=2, column=0, padx=10, pady=5, sticky="w")
wheelValue3.grid(row=2, column=1, padx=10, pady=5)
wheelLabel4.grid(row=3, column=0, padx=10, pady=5, sticky="w")
wheelValue4.grid(row=3, column=1, padx=10, pady=5)
   
# Schedule the initial update
root.after(200, update_label_text)

# Start the Tkinter event loop
root.mainloop()

