#ifndef PS4ctrl_h
#define PS4ctrl_h
class PS4ctrl {
public:
	PS4ctrl();
	bool GetPS4connected( );
	uint8_t GetBattery( );
	void SetCurveValues( float const_coeff, float x_coeff, float x2_coeff );
	uint8_t ApplyCurveValues( uint8_t value );
	void GetStickLocations( uint8_t* locations );
    uint8_t GetButtons( );
	void GetAllData( uint8_t* buffer );
	void Print( String str, bool newline );

private:
	bool updateLock;
	uint8_t LStickX;
	uint8_t LStickY;
	uint8_t RStickX;
	uint8_t RStickY;
	uint8_t Left;
	uint8_t Down;
	uint8_t Right;
	uint8_t Up;
	uint8_t Square;
	uint8_t Cross;
	uint8_t Circle;
	uint8_t Triangle;
	uint8_t L1;
	uint8_t R1;
	uint8_t L2;
	uint8_t R2;  
	float const_coeff, x_coeff, x2_coeff;   
};
#endif