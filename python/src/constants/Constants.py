# Copyright 2024
# Bryce's Senior Project
# Description: Contains all the constant values accessed elsewhere in the code

class Constants:
    
    ######################################################################################
    # DEBUG CONSTANTS
    ######################################################################################
    PS4_DEBUG = True 

    ######################################################################################
    # GENERAL MATH CONSTANTS
    ######################################################################################
    INCH_TO_MM = 25.4

    ######################################################################################
    # DRIVE CONSTANTS
    ######################################################################################
    # PS4 SPI COMMUNICATION
    PS4_SPI_CHANNEL = 0
    PS4_SPI_DEVICE = 4
    PS4_SPI_MODE = 0b01
    PS4_SPI_SPEED = 2000000

    PS4_SPI_RX_BUFFER_SIZE = 8
    PS4_SPI_TX_BUFFER_SIZE = 4

    # HEADERS FROM SPI
    SETUP_FRAME = 0
    PS4_ALL_DATA_FRAME = 1
    PS4_CONNECT_DATA_FRAME = 2
    

    # PS4 RESPONSE CURVE
    PS4_CURVE_MODE = 0
    PS4_CURVE_CONST_COEFF = 5
    PS4_CURVE_X_COEFF = 7 
    PS4_CURVE_X2_COEFF = 3

    # DRIVE BASE CHARACTERISTICS
    WHEEL_BASE_WIDTH = 22 * INCH_TO_MM
    WHEEL_BASE_LENGTH = 22 * INCH_TO_MM

    # MOTOR POWER PERCENTAGES
    MAX_NORMAL_POWER = 60
    MAX_TURBO_POWER = 80

    # SWERVE DRIVE STARTING ANGLES
    MOTOR0_BASE_ANGLE = 30
    MOTOR1_BASE_ANGLE = 20
    MOTOR2_BASE_ANGLE = 50
    MOTOR3_BASE_ANGLE = 130

    ######################################################################################
    # ARM CONSTANTS
    ######################################################################################
    # Base servo fully backward
    BASE_SERVO_MIN = 20

    # Base servo fully forward
    BASE_SERVO_MAX = 70

    # Base servo starting position
    BASE_SERVO_START = 25

    # Center servo fully closed
    CENTER_SERVO_MAX = 75

    # Center servo fully extended
    CENTER_SERVO_MIN = 25

    # Center servo starting position
    CENTER_SERVO_START = 70

    # Grabber servo fully open
    GRAB_SERVO_MIN = 50

    # Grabber servo fully closed
    GRAB_SERVO_MAX = 80

    # Grabber servo starting position
    GRAB_SERVO_START = 55

    # Grabber servo fully open
    #WRIST_SERVO_MIN = 

    # Grabber servo fully closed
    #WRIST_SERVO_MAX = 

    # Grabber servo starting position
    #WRIST_SERVO_START = 

    ######################################################################################
    # TANG NANO 9K ADDRESS CONSTANTS
    ######################################################################################
    FPGA_SPI_CHANNEL                = 0
    FPGA_SPI_DEVICE                 = 0
    FPGA_SPI_MODE                   = 0b0
    FPGA_SPI_SPEED                  = 200000

    BROADCAST_SWERVE_ADDR           = 0x01
    BROADCAST_SWERVE_ROTATION_ADDR  = 0x02			
    BROADCAST_SWERVE_DRIVE_ADDR     = 0x03
    DRIVE0_CONTROL_ADDR 			= 0x4
    DRIVE1_CONTROL_ADDR 			= 0x5
    DRIVE2_CONTROL_ADDR 			= 0x6
    DRIVE3_CONTROL_ADDR 			= 0x7

    ROTATION0_CONTROL_ADDR 			= 0x8
    ROTATION0_CURRENT_ANGLE_ADDR 	= 0x9
    ROTATION0_CURRENT_ANGLE2_ADDR 	= 0xA
    ROTATION1_CONTROL_ADDR 			= 0xB
    ROTATION1_CURRENT_ANGLE_ADDR 	= 0xC
    ROTATION1_CURRENT_ANGLE2_ADDR 	= 0xD
    ROTATION2_CONTROL_ADDR 			= 0xE
    ROTATION2_CURRENT_ANGLE_ADDR 	= 0xF
    ROTATION2_CURRENT_ANGLE2_ADDR 	= 0x10
    ROTATION3_CONTROL_ADDR 			= 0x11
    ROTATION3_CURRENT_ANGLE_ADDR 	= 0x12
    ROTATION3_CURRENT_ANGLE2_ADDR 	= 0x13

    BASE_SERVO_CONTROL_ADDR		    = 0x14
    WRIST_SERVO_CONTROL_ADDR	    = 0x15
    CENTER_SERVO_CONTROL_ADDR	    = 0x16
    GRABBER_SERVO_CONTROL_ADDR	    = 0x17
    DEBUG0_STATUS_ADDR			    = 0x18
    DEBUG1_STATUS_ADDR			    = 0x19
    DEBUG2_STATUS_ADDR			    = 0x1A
    LED_CONTROL_ADDR			    = 0x1B		

    MOTOR_CONTROL_BRAKE_BIT         = 0x7
    MOTOR_CONTROL_ENABLE_BIT        = 0x6
    MOTOR_CONTROL_DIRECTION_BIT     = 0x5
    MOTOR_CONTROL_PWM_BASE_BIT      = 0x0
    MOTOR_CONTROL_PWM_WIDTH         = 0x5

    MOTOR_CONTROL_ROTATION_BASE_BIT = 0x0
    MOTOR_CONTROL_ROTATION_WIDTH    = 0x7
    MOTOR_CONTROL_ROTATION_DIR_BIT  = 0x7

    MOTOR_STATUS_FAULT_BIT          = 0x7
    MOTOR_STATUS_TEMP_BASE_BIT      = 0x0
    MOTOR_STATUS_TEMP_WIDTH         = 0x7

    SERVO_ANGLE_BASE_BIT            = 0x0
    SERVO_ANGLE_WIDTH               = 0x8

    WR_DATA_FRAME_RW_BIT            = 0xF
    WR_DATA_FRAME_PARITY_BIT        = 0xE    
    WR_DATA_FRAME_ADDRESS_BASE_BIT  = 0x8
    WR_DATA_FRAME_ADDRESS_WIDTH     = 0x6
    WR_DATA_FRAME_DATA_BASE_BIT     = 0x0
    WR_DATA_FRAME_DATA_WIDTH        = 0x8

    RD_DATA_FRAME_RW_BIT            = 0xF
    RD_DATA_FRAME_PARITY_BIT        = 0xE
    RD_DATA_FRAME_ADDRESS_BASE_BIT  = 0x0
    RD_DATA_FRAME_ADDRESS_WIDTH     = 0x6

    BASE_SERVO_MIN			        = 5
    BASE_SERVO_MAX		            = 50
    CENTER_SERVO_MIN	            = 5
    CENTER_SERVO_MAX	            = 70
    WRIST_SERVO_MIN		            = 40
    WRIST_SERVO_MAX		            = 60
    GRABBER_SERVO_MIN		        = 97
    GRABBER_SERVO_MAX		        = 86


    ######################################################################################
    # VISION CONSTANTS
    ######################################################################################

    ######################################################################################
    # AUTO CONSTANTS
    ######################################################################################

