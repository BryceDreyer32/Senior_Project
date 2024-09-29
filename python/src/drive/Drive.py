# Copyright 2024
# Bryce's Senior Project
# Description: The class which controls the drive system
import sys
from .PS4_Ctrl import PS4_Ctrl
sys.path.append( '../constants' )
from constants.Constants import Constants

class Drive:

    def __init__( self ):
        self.runFlag = False
        self.ps4ctrl = PS4_Ctrl( Constants.PS4_SPI_CHANNEL, 
                           Constants.PS4_SPI_DEVICE, 
                           Constants.PS4_SPI_MODE )
        self.setRunFlag( )

    def setRunFlag(self):
        self.runFlag = True

    def resetRunFlag(self):
        self.runFlag = False

    def loop( self ):
        while( self.runFlag ):
            self.ps4ctrl.loop()
