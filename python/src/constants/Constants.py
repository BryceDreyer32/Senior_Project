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
    DRIVE0_CONTROL_ADDR			    = 0x04
    DRIVE0_STATUS_ADDR			    = 0x05
    DRIVE1_CONTROL_ADDR			    = 0x06
    DRIVE1_STATUS_ADDR			    = 0x07
    DRIVE2_CONTROL_ADDR			    = 0x08
    DRIVE2_STATUS_ADDR			    = 0x09
    DRIVE3_CONTROL_ADDR			    = 0x0A
    DRIVE3_STATUS_ADDR			    = 0x0B

    ROTATION0_CONTROL_ADDR		    = 0x0C
    ROTATION0_TARGET_ANGLE_ADDR	    = 0x0D
    ROTATION0_CURRENT_ANGLE_ADDR    = 0x0E
    ROTATION0_CURRENT_ANGLE2_ADDR   = 0x0F
    KP0_COEFFICIENT_ADDR		    = 0x10
    KI0_KD0_COEFFICIENTS_ADDR       = 0x11
    ROTATION0_PWM_TEST_ADDR         = 0x12

    ROTATION1_CONTROL_ADDR		    = 0x13
    ROTATION1_TARGET_ANGLE_ADDR	    = 0x14
    ROTATION1_CURRENT_ANGLE_ADDR    = 0x15
    ROTATION1_CURRENT_ANGLE2_ADDR   = 0x16
    KP1_COEFFICIENT_ADDR		    = 0x17
    KI1_KD1_COEFFICIENTS_ADDR       = 0x18
    ROTATION1_PWM_TEST_ADDR         = 0x19

    ROTATION2_CONTROL_ADDR		    = 0x1A
    ROTATION2_TARGET_ANGLE_ADDR	    = 0x1B
    ROTATION2_CURRENT_ANGLE_ADDR    = 0x1C
    ROTATION2_CURRENT_ANGLE2_ADDR   = 0x1D
    KP2_COEFFICIENT_ADDR		    = 0x1E
    KI2_KD2_COEFFICIENTS_ADDR       = 0x1F
    ROTATION2_PWM_TEST_ADDR         = 0x20

    ROTATION3_CONTROL_ADDR		    = 0x21
    ROTATION3_TARGET_ANGLE_ADDR	    = 0x22
    ROTATION3_CURRENT_ANGLE_ADDR    = 0x23
    ROTATION3_CURRENT_ANGLE2_ADDR   = 0x24
    KP3_COEFFICIENT_ADDR		    = 0x25
    KI3_KD3_COEFFICIENTS_ADDR       = 0x26
    ROTATION3_PWM_TEST_ADDR         = 0x27

    BASE_SERVO_CONTROL_ADDR			= 0x30
    WRIST_SERVO_CONTROL_ADDR		= 0x31
    CENTER_SERVO_CONTROL_ADDR	    = 0x32
    GRABBER_SERVO_CONTROL_ADDR		= 0x33
    
    DEBUG0_STATUS_ADDR			    = 0x34
    DEBUG1_STATUS_ADDR			    = 0x35
    DEBUG2_STATUS_ADDR			    = 0x36
    DEBUG3_STATUS_ADDR			    = 0x37
    LED_CONTROL_ADDR			    = 0x38

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

