#ifndef CONSTANTS_H
#define CONSTANTS_H

// DEBUG
const bool TOP_DEBUG = true;
const bool PS4_DEBUG = true;
const bool SPI_DEBUG = true;

//  GENERAL
const int SERIAL_BAUD = 115200;

// SPI
static constexpr size_t RX_BUFFER_SIZE = 4;
static constexpr size_t TX_BUFFER_SIZE = 8; // MUST BE MULTIPLE OF 8 (otherwise will send 0's!!!)
static constexpr size_t QUEUE_SIZE = 2;

// SPI DATA FRAMES
const int SETUP_FRAME = 0;
const int PS4_ALL_DATA_FRAME = 1;
const int PS4_CONNECT_DATA_FRAME = 2;


// RESPONSE CURVES


#endif