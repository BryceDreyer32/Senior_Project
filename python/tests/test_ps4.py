# Copyright 2024
# Bryce's Senior Project
# Description: 

# Copyright 2024
# Bryce's Senior Project
# Description: 

# -*- coding: utf-8 -*-
"""
Spyder Editor
This is a temporary script file.
"""
import time
import spidev
import OPi.GPIO as GPIO
spi_ch = 0
# Enable SPI
spi = spidev.SpiDev(4, spi_ch)
spi.max_speed_hz = 20000000
spi.mode = 0b01
try:
    while True:
#        spi.writebytes([0xDE, 0xAD, 0xBE, 0xEF])
        
        response = spi.readbytes(24)
        
        print("Received data:" + str(response[0]-128) + ", " + 
              str(response[1]-128) + ", " + 
              str(response[2]-128) + ", " + 
              str(response[3]-128) + ", " + 
              str(response[4]) + ", " + 
              str(response[5]) + ", " + 
              str(response[6]) + ", " + 
              str(response[7]) + ", " + 
              str(response[8]) + ", " + 
              str(response[9]) + ", " + 
              str(response[10]) + ", " + 
              str(response[11]) + ", " + 
              str(response[12]) + ", " + 
              str(response[13]) + ", " + 
              str(response[14]) + ", " + 
              str(response[15]) + ", " + 
              str(response[16]) + ", " + 
              str(response[17]) + ", " + 
              str(response[18]) + ", " + 
              str(response[19]) + ", " + 
              str(response[20]) + ", " + 
              str(response[21]) + ", " + 
              str(response[22]) + ", " + 
              str(response[23]))

except KeyboardInterrupt:
    pass
spi.close()
GPIO.cleanup()