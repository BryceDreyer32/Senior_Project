# Copyright 2024
# Bryce's Senior Project
# Description: The class which controls the Swerve Drive
import sys
sys.path.append( '../constants' )
from constants.Constants import Constants
class Swerve:

    def __init__(self ):
        # Set all motor angles to their base angle        
        self.setMotorAngle(0, Constants.MOTOR0_BASE_ANGLE)
        self.setMotorAngle(1, Constants.MOTOR1_BASE_ANGLE)
        self.setMotorAngle(2, Constants.MOTOR2_BASE_ANGLE)
        self.setMotorAngle(3, Constants.MOTOR3_BASE_ANGLE)

        # Set all motor power to 0        
        self.setMotorPower(0, 0)
        self.setMotorPower(1, 0)
        self.setMotorPower(2, 0)
        self.setMotorPower(3, 0)

    def calculateMotorPowerAngle():
        pass

    def setMotorAngle(motorId, angle):
        pass

    def setMotorPower(motorId, power):
        pass