# Copyright 2024
# Bryce's Senior Project
# Description: The class which controls the Swerve Drive
import spidev
import OPi.GPIO as GPIO
import sys
import os
#sys.path.append( '../../Constants.Constantsants' )
#from Constants.Constantsants.Constants.Constantsants import Constants.Constantsants
sys.path.append(os.path.realpath('python/src/Constants.Constants'))
import Constants

class FpgaCommunication:

    def __init__(self, spiChannel, spiDevice, spiMode, speed):
        self.runFlag = False

        # Enable SPI
        self.spi = spidev.SpiDev(spiDevice, spiChannel)
        self.spi.max_speed_hz = speed
        self.spi.mode = spiMode
        self.runFlag = True

    def fpgaWrite(self, address, data):
        # bit[15]   = 0 (write)
        # bit[14]   = parity
        # bit[13:8] = address
        # bit[7:0]  = data        
        parity = bin(((address << Constants.Constants.WR_DATA_FRAME_ADDRESS_BASE_BIT) | (data))).count('1') % 2
        msbyte = (0 << (Constants.Constants.WR_DATA_FRAME_RW_BIT-8)) | (parity << (Constants.Constants.WR_DATA_FRAME_PARITY_BIT-8)) | (address << (Constants.Constants.WR_DATA_FRAME_ADDRESS_BASE_BIT-8))
        self.spi.writebytes([msbyte, data])

    def fpgaRead(self, address):
        # First we write the read command, then the address, and data we just set to FF
        # bit[15]   = 1 (read)
        # bit[14]   = parity
        # bit[13:8] = address
        # bit[7:0]  = FF
        parity = bin(((1 << Constants.Constants.RD_DATA_FRAME_RW_BIT) | address)).count('1') % 2        
        msbyte = (1 << (Constants.Constants.RD_DATA_FRAME_RW_BIT-8)) | (parity << (Constants.Constants.RD_DATA_FRAME_PARITY_BIT-8)) | (address & 0x3F)
        self.spi.writebytes([msbyte, 0xFF])

        # Then we read back 2 bytes
        data = self.spi.readbytes(2)

        # Parity consists of a "1" for the read command, the data, and the address
        parity = (1 + bin(data[1]).count('1') + bin(address).count('1')) % 2

        #print('data[1] = ' + hex(data[1]) + ', data[0] = ' + hex(data[0]) + ', parity = ' + str(parity))

        if(data[0] == parity):
            return data[1]
        else:
            print("Parity error on SPI received read data")
            return data[1]

    def fpgaCloseConnection(self):
        self.spi.close()
        GPIO.cleanup()
        self.runFlag = False
        
