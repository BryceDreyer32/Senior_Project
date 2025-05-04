# Copyright 2024
# Bryce's Senior Project
# Description: The class which controls the PS4 
import time
import spidev
import OPi.GPIO as GPIO
import sys
sys.path.append( '../constants' )
from constants.Constants import Constants


class PS4_Ctrl:

    def __init__(self, spiChannel, spiDevice, spiMode):
        self.runFlag = False
        self.ps4Connected = False
        self.ps4Battery = False

        # Enable SPI
        self.spi = spidev.SpiDev(spiDevice, spiChannel)
        self.spi.max_speed_hz = Constants.PS4_SPI_SPEED
        self.spi.mode = spiMode

        # Check the connection with ESP32

        # Send the control curve parameters to ESP32
        self.spi.writebytes([Constants.PS4_CURVE_MODE, 
                             Constants.PS4_CURVE_CONST_COEFF, 
                             Constants.PS4_CURVE_X_COEFF, 
                             Constants.PS4_CURVE_X2_COEFF])

        # Set the run flag to be true
        self.setRunFlag()


    def print(self, str):
        if( Constants.PS4_DEBUG ):
            print( str )

    def confirmCurveSetup(self, data):
        self.print("[PS4_Ctrl.confirmCurveSetup] Confirming Curve setup" )
        # Read back the data from the ESP32 and check its correctness
        if( (data[1] == Constants.PS4_CURVE_MODE) &
            (data[2] == Constants.PS4_CURVE_CONST_COEFF) &
            (data[3] == Constants.PS4_CURVE_X_COEFF) &
            (data[4] == Constants.PS4_CURVE_X2_COEFF) ):
            correct = 1
        else:
            correct = 0

        # Then send the number of times we received the correct setup back to the ESP32 
        # four times
        self.spi.writebytes( [correct, correct, correct, correct] )

        # If the setup wasn't correct, then send it again
        if( correct == 0 ):
            self.spi.writebytes([Constants.PS4_CURVE_MODE, 
                                Constants.PS4_CURVE_CONST_COEFF, 
                                Constants.PS4_CURVE_X_COEFF, 
                                Constants.PS4_CURVE_X2_COEFF])           

    def getPs4Data(self, data):
        if(self.ps4Connected):
            self.print("[PS4_Ctrl.getPs4Data] Received data:  " + 
                str(data[1]-128).rjust(4) + ", " + 
                str(data[2]-128).rjust(4) + ", " + 
                str(data[3]-128).rjust(4) + ", " + 
                str(data[4]-128).rjust(4) + ", " + 
                str(data[5]).rjust(4))
        
    def getPs4ConnectionStatus(self, data):
        if( data[1] == 1):
            self.ps4Connected = True
        else:
            self.ps4Connected = False

        self.ps4Battery = data[2]

    def setRunFlag(self):
        self.runFlag = True

    def resetRunFlag(self):
        self.runFlag = False

    def loop(self):
        while( self.runFlag ):
            # Pull all the RX bytes (CS changes at end of readbytes - which
            # forces and EOM on the ESP side)
            data = self.spi.readbytes(Constants.PS4_SPI_RX_BUFFER_SIZE)

            self.print("[PS4_Ctrl.loop] Received data:  " + 
                str(data[0]).rjust(4) + ", " + 
                str(data[1]).rjust(4) + ", " + 
                str(data[2]).rjust(4) + ", " + 
                str(data[3]).rjust(4) + ", " + 
                str(data[4]).rjust(4) + ", " + 
                str(data[5]).rjust(4))

            # Based on the header, we'll call different routines to process the rest
            # of the data in the packet
            match ( data[0] ):
                case Constants.SETUP_FRAME              : self.confirmCurveSetup( data )
                case Constants.PS4_ALL_DATA_FRAME       : self.getPs4Data( data )
                case Constants.PS4_CONNECT_DATA_FRAME   : self.getPs4ConnectionStatus( data )
                case _                                  : self.print("Invalid header " + str(data[0]))
            
            time.sleep(0.1)

        # When the runFlag is cleared, loop is exited and cleanup performed
        self.cleanup()

    def cleanup(self):
        self.spi.close()
        GPIO.cleanup()