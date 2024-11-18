// Copyright 2024
// Bryce's Senior Project
// Description: This is the RTL for SPI

module reg_file ( 
    input           reset_n,   	     // Active low reset
    input           clock,     	     // The main clock
    input   [5:0]   address,   	     // Read / write address
    input           write_en,  	     // Write enable
    input   [7:0]   wr_data,   	     // Write data
    input           read_en,  	     // Read enable
    output reg [7:0]   rd_data,   	     // Read data
								     
    input           fault0,    	     // Fault signal from motor
    input   [6:0]   adc_temp0, 	     // Adc temperature from motor
    input           fault1,    	     // Fault signal from motor
    input   [6:0]   adc_temp1, 	     // Adc temperature from motor
    input           fault2,    	     // Fault signal from motor
    input   [6:0]   adc_temp2, 	     // Adc temperature from motor
    input           fault3,    	     // Fault signal from motor
    input   [6:0]   adc_temp3, 	     // Adc temperature from motor
    input           fault4,    	     // Fault signal from motor
    input   [6:0]   adc_temp4, 	     // Adc temperature from motor
    input           fault5,    	     // Fault signal from motor
    input   [6:0]   adc_temp5, 	     // Adc temperature from motor
    input           fault6,    	     // Fault signal from motor
    input   [6:0]   adc_temp6, 	     // Adc temperature from motor
    input           fault7,    	     // Fault signal from motor
    input   [6:0]   adc_temp7, 	     // Adc temperature from motor
								     
    output          brake0,    	     // Brake control
    output          enable0,   	     // Motor enable
    output          direction0,	     // Motor direction
    output  [4:0]   pwm0,      	     // PWM control  
    output          brake1,    	     // Brake control
    output          enable1,   	     // Motor enable
    output          direction1,	     // Motor direction
    output  [4:0]   pwm1,      	     // PWM control  
    output          brake2,    	     // Brake control
    output          enable2,   	     // Motor enable
    output          direction2,	     // Motor direction
    output  [4:0]   pwm2,      	     // PWM control  
    output          brake3,    	     // Brake control
    output          enable3,   	     // Motor enable
    output          direction3,	     // Motor direction
    output  [4:0]   pwm3,      	     // PWM control
    output          brake4,    	     // Brake control
    output          enable4,   	     // Motor enable
    output          direction4,	     // Motor direction
    output  [4:0]   pwm4,      	     // PWM control  
    output          brake5,    	     // Brake control
    output          enable5,   	     // Motor enable
    output          direction5,	     // Motor direction
    output  [4:0]   pwm5,      	     // PWM control  
    output          brake6,    	     // Brake control
    output          enable6,   	     // Motor enable
    output          direction6,	     // Motor direction
    output  [4:0]   pwm6,      	     // PWM control  
    output          brake7,    	     // Brake control
    output          enable7,   	     // Motor enable
    output          direction7,	     // Motor direction
    output  [4:0]   pwm7,      	     // PWM control  
									 
    output  [7:0]   target_angle0,   // Rotation target angle
    input   [7:0]   current_angle0,  // The current angle
    output  [7:0]   target_angle1,   // Rotation target angle
    input   [7:0]   current_angle1,  // The current angle
    output  [7:0]   target_angle2,   // Rotation target angle
    input   [7:0]   current_angle2,  // The current angle
    output  [7:0]   target_angle3,   // Rotation target angle
    input   [7:0]   current_angle3,  // The current angle

    output  [7:0]   servo_position0, // Servo 0 target position
    output  [7:0]   servo_position1, // Servo 1 target position
    output  [7:0]   servo_position2, // Servo 2 target position
    output  [7:0]   servo_position3  // Servo 3 target position
);

reg     [7:0]   reg_file    [31:0];

// Read Data is just a pointer to whatever the address is set to 
always @(posedge clock) begin
    if(read_en)
        rd_data[7:0]    <= reg_file[address];
end

/*
0x00	Reserved			                            - Writes to this address are simply dropped (don't match any of the statements below)
0x01	Broadcast to all PWM modules			        - The  "| (address == 6'h1)" in the if statements below ensures data is written to all motor drivers
0x02	Broadcast to all Swerve Rotation PWM modules	- The  "| (address == 6'h2)" ensures data is written into all rotation motor controllers
0x03	Broadcast to all Swerve Drive PWM modules       - The  "| (address == 6'h3)" writes data into all motor controllers */

// ------------- 0x4	DRIVE0_CONTROL	-------------
always @(posedge clock) begin
	if(write_en & ((address == 6'h4) | (address == 6'h3) | (address == 6'h1)))
		reg_file[4]     <=  wr_data[7:0];
end

assign brake0       = reg_file[4][7];
assign enable0      = reg_file[4][6];
assign direction0   = reg_file[4][5];
assign pwm0[4:0]    = reg_file[4][4:0]; 

// ------------- 0x5	DRIVE0_STATUS	-------------
always @(posedge clock) begin
	reg_file[5]     <=  {fault0, adc_temp0[6:0]};
end

// ------------- 0x6	DRIVE1_CONTROL	-------------
always @(posedge clock) begin
	if(write_en & ((address == 6'h6) | (address == 6'h3) | (address == 6'h1)))
		reg_file[6]     <=  wr_data[7:0];
end

assign brake1       = reg_file[6][7];
assign enable1      = reg_file[6][6];
assign direction1   = reg_file[6][5];
assign pwm1[4:0]    = reg_file[6][4:0];

// ------------- 0x7	DRIVE1_STATUS	-------------
always @(posedge clock) begin
	reg_file[7]     <=  {fault1, adc_temp1[6:0]};
end

// ------------- 0x8	DRIVE2_CONTROL	-------------
always @(posedge clock) begin
	if(write_en & ((address == 6'h8) | (address == 6'h3) | (address == 6'h1)))
		reg_file[8]     <=  wr_data[7:0];
end

assign brake2       = reg_file[8][7];
assign enable2      = reg_file[8][6];
assign direction2   = reg_file[8][5];
assign pwm2[4:0]    = reg_file[8][4:0];

// ------------- 0x9	DRIVE2_STATUS	-------------
always @(posedge clock) begin
	reg_file[9]     <=  {fault2, adc_temp2[6:0]};
end

// ------------- 0xA	DRIVE3_CONTROL	-------------
always @(posedge clock) begin
	if(write_en & ((address == 6'hA) | (address == 6'h3) | (address == 6'h1)))
		reg_file[10]     <=  wr_data[7:0];
end

assign brake3       = reg_file[10][7];
assign enable3      = reg_file[10][6];
assign direction3   = reg_file[10][5];
assign pwm3[4:0]    = reg_file[10][4:0];

// ------------- 0xB	DRIVE3_STATUS	-------------
always @(posedge clock) begin
	reg_file[11]     <=  {fault3, adc_temp3[6:0]};
end

// ------------- 0xC	ROTATION0_CONTROL	-------------
always @(posedge clock) begin
	if(write_en & ((address == 6'hC) | (address == 6'h2) | (address == 6'h1)))
		reg_file[12]     <=  wr_data[7:0];
end

assign brake4       = reg_file[12][7];
assign enable4      = reg_file[12][6];
assign direction4   = reg_file[12][5];
assign pwm4[4:0]    = reg_file[12][4:0];

// ------------- 0xD	ROTATION0_STATUS	-------------
always @(posedge clock) begin
	reg_file[13]     <=  {fault4, adc_temp4[6:0]};
end

// ------------- 0xE	ROTATION0_TARG_ANG	-------------
always @(posedge clock) begin
	if(write_en & (address == 6'hE))
		reg_file[14]     <=  wr_data[7:0];
end

assign target_angle0      = reg_file[14][7:0];

// ------------- 0xF	ROTATION0_CURR_ANG	-------------
always @(posedge clock) begin
	if(write_en & (address == 6'hF))
		reg_file[15]     <=  current_angle0[7:0];
end


// ------------- 0x10	ROTATION1_CONTROL	-------------
always @(posedge clock) begin
	if(write_en & ((address == 6'h10) | (address == 6'h2) | (address == 6'h1)))
		reg_file[16]     <=  wr_data[7:0];
end

assign brake5       = reg_file[16][7];
assign enable5      = reg_file[16][6];
assign direction5   = reg_file[16][5];
assign pwm5[4:0]    = reg_file[16][4:0];

// ------------- 0x11	ROTATION1_STATUS	-------------
always @(posedge clock) begin
	reg_file[17]     <=  {fault5, adc_temp5[6:0]};
end

// ------------- 0x12	ROTATION1_TARG_ANG	-------------
always @(posedge clock) begin
	if(write_en & (address == 6'h12))
		reg_file[18]     <=  wr_data[7:0];
end

assign target_angle0      = reg_file[18][7:0];

// ------------- 0x13	ROTATION1_CURR_ANG	-------------
always @(posedge clock) begin
	if(write_en & (address == 6'h13))
		reg_file[19]     <=  current_angle1[7:0];
end

// ------------- 0x14	ROTATION2_CONTROL	-------------
always @(posedge clock) begin
	if(write_en & ((address == 6'h14) | (address == 6'h2) | (address == 6'h1)))
		reg_file[20]     <=  wr_data[7:0];
end

assign brake6       = reg_file[20][7];
assign enable6      = reg_file[20][6];
assign direction6   = reg_file[20][5];
assign pwm6[4:0]    = reg_file[20][4:0];

// ------------- 0x15	ROTATION2_STATUS	-------------
always @(posedge clock) begin
	reg_file[21]     <=  {fault6, adc_temp6[6:0]};
end

// ------------- 0x16	ROTATION2_TARG_ANG	-------------
always @(posedge clock) begin
	if(write_en & (address == 6'h16))
		reg_file[22]     <=  wr_data[7:0];
end

assign target_angle0      = reg_file[22][7:0];

// ------------- 0x17	ROTATION2_CURR_ANG	-------------
always @(posedge clock) begin
	if(write_en & (address == 6'h17))
		reg_file[23]     <=  current_angle2[7:0];
end

// ------------- 0x18	ROTATION3_CONTROL	-------------
always @(posedge clock) begin
	if(write_en & ((address == 6'h18) | (address == 6'h2) | (address == 6'h1)))
		reg_file[24]     <=  wr_data[7:0];
end

assign brake7       = reg_file[24][7];
assign enable7      = reg_file[24][6];
assign direction7   = reg_file[24][5];
assign pwm7[4:0]    = reg_file[24][4:0];

// ------------- 0x19	ROTATION3_STATUS	-------------
always @(posedge clock) begin
	reg_file[25]     <=  {fault7, adc_temp7[6:0]};
end

// ------------- 0x1A	ROTATION3_TARG_ANG	-------------
always @(posedge clock) begin
	if(write_en & (address == 6'h1A))
		reg_file[26]     <=  wr_data[7:0];
end

assign target_angle0      = reg_file[26][7:0];

// ------------- 0x1B	ROTATION3_CURR_ANG	-------------
always @(posedge clock) begin
	if(write_en & (address == 6'h1B))
		reg_file[27]     <=  current_angle3[7:0];
end

// --------------- 0x1C	SERVO0_CONTROL	----------------
always @(posedge clock) begin
	if(write_en & (address == 6'h1C))
		reg_file[28]     <=  wr_data[7:0];
end

assign servo_position0    = reg_file[28][7:0];

// --------------- 0x1D	SERVO0_CONTROL	----------------
always @(posedge clock) begin
	if(write_en & (address == 6'h1D))
		reg_file[29]     <=  wr_data[7:0];
end

assign servo_position1    = reg_file[29][7:0];

// --------------- 0x1E	SERVO0_CONTROL	----------------
always @(posedge clock) begin
	if(write_en & (address == 6'h1E))
		reg_file[30]     <=  wr_data[7:0];
end

assign servo_position2    = reg_file[30][7:0];

// --------------- 0x1F	SERVO0_CONTROL	----------------
always @(posedge clock) begin
	if(write_en & (address == 6'h1F))
		reg_file[31]     <=  wr_data[7:0];
end

assign servo_position3    = reg_file[31][7:0];
			

endmodule