# Copyright 2024
# Bryce's Senior Project
# Description: This is the main entry point of the Robot code
import time
from multiprocessing import Process
from drive.Drive import Drive 

class MainLoop:
    # Defining main (setup) function 
    def __init__( self ):
        # Create an instance of each of the classes that we will assign to a process
        self.drive = Drive()
        self.drive.loop()

        # Create a new process for each of the instances, and assign the loop of
        # each to the process
        drvProcess = Process( target = self.drive.loop )
        armProcess = Process( )
        visionProcess = Process( )

        # Start each process
        drvProcess.start( )

        # Call the main loop
        self.loop()

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
