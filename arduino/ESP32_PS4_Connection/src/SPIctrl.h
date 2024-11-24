#ifndef SPIctrl_h
#define SPIctrl_h
#include "constants.h"

class SPIctrl {
public:
	SPIctrl();
    void GetSetup( );
    void ConfirmSetup();
    void SendData( uint8_t* buffer, uint8_t buf_size );
    void Print( String str, bool newline );
    void PrintTxBuffer( uint8_t* buffer, bool newline );
    float const_coeff, x_coeff, x2_coeff;   


private:
    uint8_t tx_buf[TX_BUFFER_SIZE] {0, 0, 0, 0, 0, 0, 0, 0};
    uint8_t rx_buf[RX_BUFFER_SIZE] {0, 0, 0, 0};
    uint8_t curve_mode;
    char msgString[25];
    char messageString[200];
};
#endif
