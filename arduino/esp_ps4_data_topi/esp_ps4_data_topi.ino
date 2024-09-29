#include <ESP32SPISlave.h>
#include "helper.h"
#include <PS4Controller.h>
unsigned long lastTimeStamp = 0;
ESP32SPISlave slave;
static constexpr size_t RX_BUFFER_SIZE = 4;
static constexpr size_t TX_BUFFER_SIZE = 24;
static constexpr size_t QUEUE_SIZE = 4;
uint8_t tx_buf[TX_BUFFER_SIZE];
uint8_t rx_buf[RX_BUFFER_SIZE] {0, 0, 0, 0};
uint8_t curve_mode;
float const_coeff, x_coeff, x2_coeff;   
char msgString[25];
char messageString[200];

void notify()
{
  sprintf(messageString, "%4d,%4d,%4d,%4d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d,%3d",
  PS4.LStickX(),
  PS4.LStickY(),
  PS4.RStickX(),
  PS4.RStickY(),
  PS4.Left(),
  PS4.Down(),
  PS4.Right(),
  PS4.Up(),
  PS4.Square(),
  PS4.Cross(),
  PS4.Circle(),
  PS4.Triangle(),
  PS4.L1(),
  PS4.R1(),
  PS4.L2(),
  PS4.R2(),  
  PS4.Share(),
  PS4.Options(),
  PS4.PSButton(),
  PS4.Touchpad(),
  PS4.Charging(),
  PS4.Audio(),
  PS4.Mic(),
  PS4.Battery());
  tx_buf[0] = PS4.LStickX()+128;
  tx_buf[1] = PS4.LStickY()+128;
  tx_buf[2] = PS4.RStickX()+128;
  tx_buf[3] = PS4.RStickY()+128;
  tx_buf[4] = PS4.Left();
  tx_buf[5] = PS4.Down();
  tx_buf[6] = PS4.Right();
  tx_buf[7] = PS4.Up();
  tx_buf[8] = PS4.Square();
  tx_buf[9] = PS4.Cross();
  tx_buf[10] = PS4.Circle();
  tx_buf[11] = PS4.Triangle();
  tx_buf[12] = PS4.L1();
  tx_buf[13] = PS4.R1();
  tx_buf[14] = PS4.L2();
  tx_buf[15] = PS4.R2();  
  tx_buf[16] = PS4.Share();
  tx_buf[17] = PS4.Options();
  tx_buf[18] = PS4.PSButton();
  tx_buf[19] = PS4.Touchpad();
  tx_buf[20] = PS4.Charging();
  tx_buf[21] = PS4.Audio();
  tx_buf[22] = PS4.Mic();
  tx_buf[23] = PS4.Battery();
  //Only needed to print the message properly on serial monitor. Else we dont need it.
  if (millis() - lastTimeStamp > 50)
  {
    lastTimeStamp = millis();
    Serial.println(messageString);
    slave.queue(tx_buf, NULL, TX_BUFFER_SIZE);
  }
}
void onConnect()
{
  Serial.println("Connected!.");
}
void onDisConnect()
{
  Serial.println("Disconnected!.");    
}
void setup()
{
    Serial.begin(115200);
    delay(2000);
    // SPI connection
    slave.setDataMode(SPI_MODE0);   // default: SPI_MODE0
    slave.setQueueSize(QUEUE_SIZE); // default: 1, requres 2 in this example
    slave.begin();  // default: HSPI (please refer README for pin assignments)
    Serial.println("Initialize SPI tx/rx buffers");
//    initializeBuffers(tx_buf, rx_buf, TX_BUFFER_SIZE, RX_BUFFER_SIZE);
    initializeRxBuffers(rx_buf, RX_BUFFER_SIZE);

    Serial.println("SPI Slave Started");

         // Receive curve setup from Pi
    slave.queue(NULL, rx_buf, RX_BUFFER_SIZE);
    const std::vector<size_t> received_bytes = slave.wait();
    curve_mode = rx_buf[0];
    const_coeff = float(-rx_buf[1]);
    x_coeff = float(float(rx_buf[2]) / 10);
    x2_coeff = float(float(rx_buf[3]) / 1000);
    Serial.print("curve mode = ");
    Serial.println(curve_mode);
    sprintf(msgString, "Polynomial = %4f + %4fx + %4fx^2", const_coeff, x_coeff, x2_coeff);
    Serial.println(msgString);

    // PS4 connection
    PS4.attach(notify);
    PS4.attachOnConnect(onConnect);
    PS4.attachOnDisconnect(onDisConnect);
    PS4.begin();
    Serial.println("Waiting for PS4 connection...");
}
void loop()
{
}