import os, sys
sys.path.append(os.path.realpath('python/src/constants'))
sys.path.append(os.path.realpath('python/src/subsystem/fpga'))
import Constants

class HAL:
    """Hardware Abstraction Layer for Onyx / Helios Robot code"""

    def __init__(self, fpga):
        """Constructor needs a pointer to the FPGA object, which has the SPI interface
        that we will interact through."""
        self.fpga = fpga


    def run_motor(self, motor, speed, direction):
        """Starts running the motor with the given index, in the specified direction
        with the given speed"""
        start_angle = self.get_angle(motor)
        val = ((1 << 7) | (direction << 6) | (speed & 0x3F))
        if(motor == 0):
            self.fpga.fpgaWrite(Constants.Constants.ROTATION0_CONTROL_ADDR, val)

        elif(motor == 1):
            self.fpga.fpgaWrite(Constants.Constants.ROTATION1_CONTROL_ADDR, val)

        elif(motor == 2):
            self.fpga.fpgaWrite(Constants.Constants.ROTATION2_CONTROL_ADDR, val)

        elif(motor == 3):
            self.fpga.fpgaWrite(Constants.Constants.ROTATION3_CONTROL_ADDR, val)


    def stop_motor(self, motor):
        """Stops the motor with the given index"""
        if(motor == 0):
            self.fpga.fpgaWrite(Constants.Constants.ROTATION0_CONTROL_ADDR, 0x0)

        elif(motor == 1):
            self.fpga.fpgaWrite(Constants.Constants.ROTATION1_CONTROL_ADDR, 0x0)

        elif(motor == 2):
            self.fpga.fpgaWrite(Constants.Constants.ROTATION2_CONTROL_ADDR, 0x0)

        elif(motor == 3):
            self.fpga.fpgaWrite(Constants.Constants.ROTATION3_CONTROL_ADDR, 0x0)


    def get_angle(self, motor):
        """Retrieves the raw angle measurement of a swerve module (4096 possible values)"""
        if(motor == 0):
            self.fpga.fpgaWrite(Constants.Constants.ROTATION0_CURRENT_ANGLE2_ADDR, 0x40)
            lsb_data = self.fpga.fpgaRead(Constants.Constants.ROTATION0_CURRENT_ANGLE_ADDR)
            msb_data = self.fpga.fpgaRead(Constants.Constants.ROTATION0_CURRENT_ANGLE2_ADDR)

        elif(motor == 1):
            self.fpga.fpgaWrite(Constants.Constants.ROTATION1_CURRENT_ANGLE2_ADDR, 0x40)
            lsb_data = self.fpga.fpgaRead(Constants.Constants.ROTATION1_CURRENT_ANGLE_ADDR)
            msb_data = self.fpga.fpgaRead(Constants.Constants.ROTATION1_CURRENT_ANGLE2_ADDR)

        elif(motor == 2):
            self.fpga.fpgaWrite(Constants.Constants.ROTATION2_CURRENT_ANGLE2_ADDR, 0x40)
            lsb_data = self.fpga.fpgaRead(Constants.Constants.ROTATION2_CURRENT_ANGLE_ADDR)
            msb_data = self.fpga.fpgaRead(Constants.Constants.ROTATION2_CURRENT_ANGLE2_ADDR)

        elif(motor == 3):
            self.fpga.fpgaWrite(Constants.Constants.ROTATION3_CURRENT_ANGLE2_ADDR, 0x40)
            lsb_data = self.fpga.fpgaRead(Constants.Constants.ROTATION3_CURRENT_ANGLE_ADDR)
            msb_data = self.fpga.fpgaRead(Constants.Constants.ROTATION3_CURRENT_ANGLE2_ADDR)

        rd_data = ((msb_data << 8) | lsb_data) & 0xFFF
        return rd_data
