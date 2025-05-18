# Copyright 2024
# Bryce's Senior Project
# Description: This is the main entry point of the Robot code
import time, sys, os
from PyQt5 import QtWidgets, uic
from multiprocessing import Process
from drive.Drive import Drive
from gui.Gui import Gui 
print(os.getcwd())
sys.path.append(os.path.realpath('python/src/constants'))
sys.path.append(os.path.realpath('python/src/subsystem/fpga'))
import Constants
import FpgaCommunicationGui

class MainLoop:
    # Defining main (setup) function 
    def __init__( self ):
        # FPGA instance
#        fpga = FpgaCommunicationGui.FpgaCommunicationGui(self, Constants.Constants.FPGA_SPI_CHANNEL, Constants.Constants.FPGA_SPI_DEVICE, Constants.Constants.FPGA_SPI_MODE, Constants.Constants.FPGA_SPI_SPEED)

        # Create an instance of each of the classes that we will assign to a process
        self.drive = Drive()

#        # Create a new process for each of the instances, and assign the loop of
#        # each to the process
#        drvProcess = Process( target = self.drive.loop )
#        armProcess = Process( )
#        visionProcess = Process( )

        # Start each process
#        drvProcess.start( )

        # Start the GUI process
        app = QtWidgets.QApplication(sys.argv)
        gui = Gui()

        gui.show()
        sys.exit(app.exec_())

        # Call the main loop
#        self.loop()

    def cleanup( self ):
        # Clear all the run flags in the processes
        self.drive.resetRunFlag()

        # Wait 1 second to give everything time to clear out, and then exit
        time.sleep( 1 )

    # The main loop
    def loop( self ):
        try:
            while True:
                time.sleep( 0.1 )
        except KeyboardInterrupt:
            print( 'Keyboard Interrupted' )
            self.cleanup()
            print( 'Cleanup complete, exiting' )


# Using the special variable  
# __name__ 
if __name__=="__main__": 
    main = MainLoop()
