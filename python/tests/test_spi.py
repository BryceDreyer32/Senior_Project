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
spi = spidev.SpiDev(0, spi_ch)
spi.max_speed_hz = 20000000
spi.mode = 0b01
try:
    while True:
        spi.writebytes([0xDE, 0xAD, 0xBE, 0xEF])
        spi.readbytes(2)
        
except KeyboardInterrupt:
    pass
spi.close()
GPIO.cleanup()