// Copyright 2024
// Bryce's Senior Project
// Description: Takes commands through the UART (from FPGA), and creates
// the required PWM signal for the motor controller.
// Also monitors the temperature of the motor (and motor controller, maybe?)

#include <Arduino.h>
#include "Functions.h"

// Pin declarations
const int MOTOR_ENABLE_PIN    = D3;
const int MOTOR_DIRECTION_PIN = D5; 
const int MOTOR_BRAKE_N_PIN   = D1;
const int MOTOR_PWM_PIN       = D2;
const int MOTOR_FAULT_PIN     = D0;
const int MOTOR_ADC_TEMP_PIN  = A0;

const int SERIAL_BAUD         = 115200;
// const int HW_SERIAL_BAUD      = 115200;

// Constant system parameters
const int MAX_MOTOR_TEMP      = 60;
const int MAX_SYSTEM_TEMP     = 85;
const int PWM_FREQUENCY       = 45000; // 45kHz
const int CHECK_TEMP_FREQ     = 20;

// Data frame constants
const uint8 BRAKE_BIT         = 7;
const uint8 ENABLE_BIT        = 6;
const uint8 DIRECTION_BIT     = 5;

// Variables
uint8_t brakeStatus  = 0;
uint8_t enableStatus = 0;
uint8_t dirStatus    = 0;
uint8_t pwmStatus    = 0;
int motorTemperature = 0;
int incomingByte     = 0;
int checkTempCount   = 0;

// HardwareSerial hwSerial(2);

void setup() {
   // Set the pin directions
   pinMode(MOTOR_ENABLE_PIN, OUTPUT);
   pinMode(MOTOR_DIRECTION_PIN, OUTPUT);
   pinMode(MOTOR_BRAKE_N_PIN, OUTPUT);
   pinMode(MOTOR_PWM_PIN, OUTPUT);
   pinMode(MOTOR_FAULT_PIN, INPUT);
   pinMode(MOTOR_ADC_TEMP_PIN, INPUT);

   // Set the initial values
   digitalWrite(MOTOR_ENABLE_PIN, LOW);
   digitalWrite(MOTOR_DIRECTION_PIN, LOW);
   digitalWrite(MOTOR_BRAKE_N_PIN, LOW);
   digitalWrite(MOTOR_PWM_PIN, LOW);

   // Set the frequency of PWM
   analogWriteFreq(PWM_FREQUENCY);
   analogWriteRange(32);

   // Setup the UART to 115200 baud rate
   // hwSerial.begin(HW_SERIAL_BAUD, SERIAL_8E2);//, RXD2, TXD2);

   Serial.begin(SERIAL_BAUD, SERIAL_8E2);
   Serial.print("Starting up...");  
}

void loop() {
   if (Serial.available() > 0) {
      // Read the incoming byte:
      incomingByte = Serial.read();

      Serial.print("Received ");
      Serial.println(incomingByte);
      
      // Update the brake, enable, direction, and PWM values
      processByte(incomingByte);
   }

   // Periodically check the temperature on the motor and motor board
   if(checkTempCount == CHECK_TEMP_FREQ) {
      checkTempCount = 0;
      if(checkMotorTempOkay())
         Serial.println("Temperature okay");
      else {
         // If motor temp or system temp too high, shutdown motor driver
         digitalWrite(MOTOR_ENABLE_PIN, LOW);
         Serial.print("Shutdown");
      }
   }
   else
      checkTempCount++;

   delay(100);

}

bool checkMotorTempOkay() {
   motorTemperature = analogRead(A0);
   
   if(motorTemperature > MAX_MOTOR_TEMP)
      return false;
   else 
      return true;
}

bool checkSystemTempOkay() { return true; }
   
void processByte(uint8_t incomingByte) {
   // if(incomingByte == 10)
   //   return;

   enableStatus   = (incomingByte >> ENABLE_BIT) & 0x01;
   brakeStatus    = (incomingByte >> BRAKE_BIT) & 0x01;
   dirStatus      = (incomingByte >> DIRECTION_BIT) & 0x01;
   pwmStatus      = incomingByte & 0x1F;
   
   Serial.print("enable = ");
   Serial.println(enableStatus);
   Serial.print("brake = ");
   Serial.println(brakeStatus);
   Serial.print("direction = ");
   Serial.println(dirStatus);
   Serial.print("pwm = ");
   Serial.println(pwmStatus);


   // Write the value to the PWM and the direction before enabling
   digitalWrite(MOTOR_BRAKE_N_PIN, brakeStatus);
   digitalWrite(MOTOR_DIRECTION_PIN, dirStatus);
   analogWrite(MOTOR_PWM_PIN, pwmStatus);

   // And then write the enable
   digitalWrite(MOTOR_ENABLE_PIN, enableStatus);

}
