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
    MAX_NORMAL_POWER = 80
    MAX_TURBO_POWER = 90

    ######################################################################################
    # ARM CONSTANTS
    ######################################################################################

    ######################################################################################
    # VISION CONSTANTS
    ######################################################################################

    ######################################################################################
    # AUTO CONSTANTS
    ######################################################################################
