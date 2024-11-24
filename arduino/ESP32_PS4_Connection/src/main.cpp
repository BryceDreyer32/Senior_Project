// Copyright 2024
// Bryce's Senior Project
// Description: This is the main entry point of the ESP32 code

#include <Arduino.h>
#include "PS4ctrl.h"
#include "SPIctrl.h"

SPIctrl *spictrl;
PS4ctrl *ps4ctrl;
uint8_t buf[6];
char msgStr[30];
bool ps4wasconn, ps4isconn;

void Print( String str ) {
    if( TOP_DEBUG )
        Serial.println(str);
}

void setup() {
    // UART setup
    Serial.begin( 115200 );
    delay( 1000 );

    // SPI setup
    Print( "[MAIN] Setting up SPI and PS4 parameters" );
    spictrl = new SPIctrl();
    spictrl->GetSetup();
    spictrl->ConfirmSetup();

    // PS4 connect
    Print( "[MAIN] Setting up PS4 controller" );
    ps4wasconn = false;
    ps4isconn = false;
    ps4ctrl = new PS4ctrl();
    ps4ctrl->SetCurveValues(spictrl->const_coeff, spictrl->x_coeff, spictrl->x2_coeff);

    Print( "[MAIN] Setup done, starting main loop" );

}

void loop() {
    // Check if PS4 connected
    ps4isconn = ps4ctrl->GetPS4connected();
    
    // If PS4 is connected, but wasn't then it's a new connection
    // So we need to let the Pi know
    if( ps4isconn & !ps4wasconn ) {
        uint8_t buf[8] {PS4_CONNECT_DATA_FRAME, 1, ps4ctrl->GetBattery(), 0, 0, 0, 0, 0};
        spictrl->SendData( buf, TX_BUFFER_SIZE );
    }
    // Otherwise if we have a stable connection, then send the latest data
    else if( ps4isconn & ps4wasconn ) {
        ps4ctrl->GetAllData( buf );
        spictrl->SendData( buf, TX_BUFFER_SIZE );        
    }
    // If the connection has dropped, then send info
    else if( !ps4isconn & ps4wasconn ) {
        uint8_t buf[8] {PS4_CONNECT_DATA_FRAME, 0, 0, 0, 0, 0, 0, 0};
        spictrl->SendData( buf, TX_BUFFER_SIZE );
    }
    // If we're aren't connected, and haven't been, then just idle
    else {
        delay(100);
    }

    // Update the was-connected with the is-connected
    ps4wasconn = ps4isconn;
}

