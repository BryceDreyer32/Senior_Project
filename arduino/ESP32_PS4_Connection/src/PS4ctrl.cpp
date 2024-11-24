// Copyright 2024
// Bryce's Senior Project
// Description: PS4 controller functions

#include <Arduino.h>
#include "PS4ctrl.h"
#include "constants.h"
#include <PS4Controller.h>

char messageString[200];
char msgString[30];

PS4ctrl::PS4ctrl() {
    // PS4 connection
    PS4.begin();
    Print( "[PS4ctrl:PS4ctrl] Waiting for PS4 connection...", true );
    while( !PS4.isConnected() ) {
      delay(100);
      //Print(".");
    }
    Print( "[PS4ctrl:PS4ctrl] PS4 connected!", true );

}

bool PS4ctrl::GetPS4connected() {
    return PS4.isConnected();
}

uint8_t PS4ctrl::GetBattery( ) {
    return uint8_t(PS4.Battery());
}

void PS4ctrl::SetCurveValues( float const_val, float x_val, float x2_val ) {
    const_coeff = const_val;
    x_coeff = x_val;
    x2_coeff = x2_val;
}

uint8_t PS4ctrl::ApplyCurveValues( uint8_t value ) {
    return uint8_t( const_coeff + (x_coeff * value) + (x2_coeff * value * value));
} 

void PS4ctrl::GetStickLocations( uint8_t* locations ) {
    locations[0] = 0;
    locations[1] = ApplyCurveValues(PS4.LStickX())+128;
    locations[2] = ApplyCurveValues(PS4.LStickY())+128;
    locations[3] = ApplyCurveValues(PS4.RStickX())+128;
    locations[4] = ApplyCurveValues(PS4.RStickY())+128;
    locations[5] = 0;
    locations[6] = 0;
    locations[7] = 0;
}

uint8_t PS4ctrl::GetButtons( ) {
    return uint8_t( PS4.Square()   << 0 |
                    PS4.Cross()    << 1 |
                    PS4.Circle()   << 2 |
                    PS4.Triangle() << 3 |
                    PS4.L1()       << 4 |
                    PS4.L2()       << 5 |
                    PS4.R1()       << 6 |
                    PS4.R2()       << 7 );
}

void PS4ctrl::GetAllData( uint8_t* buffer ) {
    // Put the stick data in [4:1], overwrite [0] with the all-data header, and then add the buttons into [5]
    GetStickLocations( buffer );
    buffer[0] = uint8_t(PS4_ALL_DATA_FRAME);
    buffer[5] = GetButtons();
    buffer[6] = 0;
    buffer[7] = 0;

  //sprintf( msgString, "%4d,%4d,%4d,%4d,%4d,%4d", buffer[0], buffer[1], buffer[2], buffer[3], buffer[4], buffer[5] );
  //Print( msgString, true );

}

void PS4ctrl::Print( String str, bool newline ) {
    if( PS4_DEBUG )
        if( newline )
            Serial.println( str );
        else
            Serial.print( str );
}