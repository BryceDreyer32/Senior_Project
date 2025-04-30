// Copyright 2024
// Bryce's Senior Project
// Description: SPI functions

#include <Arduino.h>
#include "SPIctrl.h"
#include "helper.h"
#include <ESP32SPISlave.h>

ESP32SPISlave slave;

SPIctrl::SPIctrl() {    
    // SPI connection
    slave.setDataMode(SPI_MODE0);   // default: SPI_MODE0
    slave.setQueueSize(QUEUE_SIZE); // default: 1, requres 2 in this example
    slave.begin(VSPI);  // default: HSPI (please refer README for pin assignments)
    Print( "Initialize SPI tx/rx buffers", true );
    initializeBuffers(tx_buf, rx_buf, TX_BUFFER_SIZE, RX_BUFFER_SIZE);
    //initializeRxBuffers(rx_buf, RX_BUFFER_SIZE);

    if( SPI_DEBUG )
        Print( "SPI Slave Started", true );

}

void SPIctrl::GetSetup() {
    // Receive curve setup from Pi
    slave.queue(NULL, rx_buf, RX_BUFFER_SIZE);
    const std::vector<size_t> received_bytes = slave.wait();
    curve_mode = rx_buf[0];
    const_coeff = float(-rx_buf[1]);
    x_coeff = float(float(rx_buf[2]) / 10);
    x2_coeff = float(float(rx_buf[3]) / 1000);

    if(SPI_DEBUG) {
        Print( "[SPIctrl.GetSetup] curve mode = ", false );
        Print( String(curve_mode), true );
        sprintf(msgString, "[SPIctrl.GetSetup] Polynomial = %4f + %4fx + %4fx^2", const_coeff, x_coeff, x2_coeff);
        Print( msgString, true );
    }
}

void SPIctrl::ConfirmSetup() {
    // Send a SETUP_FRAME with the data back to the Pi to be checked
    Print( "[SPIctrl.ConfirmSetup] Sending back the setup data for confirmation ", true );
    uint8_t buf[8] {SETUP_FRAME, curve_mode, uint8_t(-const_coeff), uint8_t(x_coeff*10), uint8_t(x2_coeff*1000), 0, 0, 0};
    Print( "[SPIctrl.ConfirmSetup] Sending : ", false);
    PrintTxBuffer( buf, true );
    
    SendData( buf, TX_BUFFER_SIZE );

    // Wait for Pi to send back 4 data bytes
    Print( "[SPIctrl.ConfirmSetup] Waiting for Pi to send 4 bytes ", true );
    slave.queue(NULL, rx_buf, RX_BUFFER_SIZE);
    const std::vector<size_t> received_bytes = slave.wait();

    // Sum up the result of the response, if it's >=3, then we received the correct setup
    uint8_t sum = rx_buf[0] + rx_buf[1] + rx_buf[2] + rx_buf[3];

    Print( "[SPIctrl.ConfirmSetup] Response from PI summed up to ", false );
    Print( String(sum), true );

    // If we get <3, then we got a bad setup, so Pi will send again
    // and we will do a recursive call of this function to check it
    if( sum < 3 ) {
        GetSetup();
        ConfirmSetup();
    }

    // Once we have a good setup, we will exit this function
    Print( "[SPIctrl.ConfirmSetup] Setup confirmed, proceeding...", true );
}

void SPIctrl::SendData( uint8_t* buffer, uint8_t buf_size ){
    Print( "[SPIctrl.SendData] Sending {", false );
    for(int i = 0; i < buf_size-1; i++) {
        Print(String(buffer[i]), false);
        Print(", ", false);
    }
    Print(String(buffer[buf_size-1]), false);
    Print("}", true);

    slave.queue( buffer, NULL, buf_size );
    const std::vector<size_t> received_bytes = slave.wait();

}

void SPIctrl::Print( String str, bool newline ) {
    if( SPI_DEBUG )
        if( newline )
            Serial.println( str );
        else
            Serial.print( str );
}

void SPIctrl::PrintTxBuffer( uint8_t* buffer, bool newline ) {
    if( SPI_DEBUG )
        Serial.print( "{" );
        for(int i = 0; i < TX_BUFFER_SIZE - 1 ; i++) {
            Serial.print( buffer[i] );
            Serial.print( "," );
        }        
        Serial.print( buffer[TX_BUFFER_SIZE - 1] );

        if( newline )
            Serial.println( "}" );
        else
            Serial.print( "}" );

}