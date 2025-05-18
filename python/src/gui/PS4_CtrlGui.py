# Copyright 2024
# Bryce's Senior Project
# Description: The class which controls the PS4 
import time
import spidev
import OPi.GPIO as GPIO
import sys
sys.path.append( '../constants' )
from Constants import Constants
from PyQt5.QtCore import QThread, pyqtSignal

class PS4_CtrlGui(QThread):
    ps4_data = pyqtSignal(str)

    def __init__(self):            
        super().__init__() 
        self.runFlag = False
        self.ps4Connected = False
        self.ps4Battery = False

        # Enable SPI
        self.spi = spidev.SpiDev(Constants.PS4_SPI_DEVICE, Constants.PS4_SPI_CHANNEL)
        self.spi.max_speed_hz = Constants.PS4_SPI_SPEED
        self.spi.mode = Constants.PS4_SPI_MODE

        # Check the connection with ESP32

        # Send the control curve parameters to ESP32
        self.spi.writebytes([Constants.PS4_CURVE_MODE, 
                             Constants.PS4_CURVE_CONST_COEFF, 
                             Constants.PS4_CURVE_X_COEFF, 
                             Constants.PS4_CURVE_X2_COEFF])

        # Set the run flag to be true
        self.setRunFlag()

    def setRunFlag(self):
        self.runFlag = True

    def resetRunFlag(self):
        self.runFlag = False

    def confirmCurveSetup(self, data):
        #print("[PS4_Ctrl.confirmCurveSetup] Confirming Curve setup" )
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


    def run(self):
        while( self.runFlag ):
            # Pull all the RX bytes (CS changes at end of readbytes - which
            # forces and EOM on the ESP side)
            data = self.spi.readbytes(Constants.PS4_SPI_RX_BUFFER_SIZE)
            
            if(data[0] == Constants.SETUP_FRAME ):
                self.confirmCurveSetup( data )
            else:
                dataStr = ""
                for num in data:
                    dataStr += str(num)+","

                self.ps4_data.emit(dataStr[:-1])

                time.sleep(0.2)

        # When the runFlag is cleared, loop is exited and cleanup performed
        self.cleanup()

    def cleanup(self):
        self.spi.close()
        GPIO.cleanup()