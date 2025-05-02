// Copyright 2024
// Bryce's Senior Project
// Description: This is the RTL for Address Decoder and Register File

module reg_file ( 
    input           reset_n,   	     // Active low reset
    input           clock,     	     // The main clock
    input   [5:0]   address,   	     // Read / write address
    input           write_en,  	     // Write enable
    input   [7:0]   wr_data,   	     // Write data
    input           read_en,  	     // Read enable
    output reg [7:0]   rd_data,   	 // Read data
								     
    // DRIVE MOTORS
    input           fault0,    	     // Fault signal from motor
    input   [5:0]   adc_temp0, 	     // Adc temperature from motor
    input           fault1,    	     // Fault signal from motor
    input   [5:0]   adc_temp1, 	     // Adc temperature from motor
    input           fault2,    	     // Fault signal from motor
    input   [5:0]   adc_temp2, 	     // Adc temperature from motor
    input           fault3,    	     // Fault signal from motor
    input   [5:0]   adc_temp3, 	     // Adc temperature from motor
    input           fault4,    	     // Fault signal from motor
    input   [5:0]   adc_temp4, 	     // Adc temperature from motor
    input           fault5,    	     // Fault signal from motor
    input   [5:0]   adc_temp5, 	     // Adc temperature from motor
    input           fault6,    	     // Fault signal from motor
    input   [5:0]   adc_temp6, 	     // Adc temperature from motor
    input           fault7,    	     // Fault signal from motor
    input   [5:0]   adc_temp7, 	     // Adc temperature from motor
								     
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
    output          brake5,    	     // Brake control
    output          enable5,   	     // Motor enable
    output          direction5,	     // Motor direction 
    output          brake6,    	     // Brake control
    output          enable6,   	     // Motor enable
    output          direction6,	     // Motor direction
    output          brake7,    	     // Brake control
    output          enable7,   	     // Motor enable
    output          direction7,	     // Motor direction 

    input  [7:0]    startup_fail,

    // ROTATION MOTORS
    output              enable_stall_chk0,   // Enable the stall check
    output      [7:0]   kp0,                 // Proportional Constant: fixed point 4.4
    output      [3:0]   ki0,                 // Integral Constant: fixed point 0.4
    output      [3:0]   kd0,                 // Derivative Constant: fixed point 0.4
    output              rot_pwm_ovrd0,       // Rotation motor override enable
    output              pwm_dir_ovrd0,       // Rotation motor override direction
    output      [5:0]   pwm_ratio_ovrd0,     // Rotation motor override value

    output              enable_stall_chk1,   // Enable the stall check
    output      [7:0]   kp1,                 // Proportional Constant: fixed point 4.4
    output      [3:0]   ki1,                 // Integral Constant: fixed point 0.4
    output      [3:0]   kd1,                 // Derivative Constant: fixed point 0.4
    output              rot_pwm_ovrd1,       // Rotation motor override enable
    output              pwm_dir_ovrd1,       // Rotation motor override direction
    output      [5:0]   pwm_ratio_ovrd1,     // Rotation motor override value

    output              enable_stall_chk2,   // Enable the stall check
    output      [7:0]   kp2,                 // Proportional Constant: fixed point 4.4
    output      [3:0]   ki2,                 // Integral Constant: fixed point 0.4
    output      [3:0]   kd2,                 // Derivative Constant: fixed point 0.4
    output              rot_pwm_ovrd2,       // Rotation motor override enable
    output              pwm_dir_ovrd2,       // Rotation motor override direction
    output      [5:0]   pwm_ratio_ovrd2,     // Rotation motor override value

    output              enable_stall_chk3,   // Enable the stall check
    output      [7:0]   kp3,                 // Proportional Constant: fixed point 4.4
    output      [3:0]   ki3,                 // Integral Constant: fixed point 0.4
    output      [3:0]   kd3,                 // Derivative Constant: fixed point 0.4
    output              rot_pwm_ovrd3,       // Rotation motor override enable
    output              pwm_dir_ovrd3,       // Rotation motor override direction
    output      [5:0]   pwm_ratio_ovrd3,     // Rotation motor override value

    output      [11:0]  target_angle0,      // Rotation target angle
    input       [11:0]  current_angle0,     // The current angle
    output      [11:0]  target_angle1,      // Rotation target angle
    input       [11:0]  current_angle1,     // The current angle
    output      [11:0]  target_angle2,      // Rotation target angle
    input       [11:0]  current_angle2,     // The current angle
    output      [11:0]  target_angle3,      // Rotation target angle
    input       [11:0]  current_angle3,     // The current angle
    output  reg         update_angle0,      // Start rotation to angle
    output  reg         update_angle1,      // Start rotation to angle
    output  reg         update_angle2,      // Start rotation to angle
    output  reg         update_angle3,      // Start rotation to angle
    output  reg         abort_angle0,       // Aborts rotating to angle
    output  reg         abort_angle1,       // Aborts rotating to angle
    output  reg         abort_angle2,       // Aborts rotating to angle
    output  reg         abort_angle3,       // Aborts rotating to angle
    input               angle_done0,        // Arrived at target angle
    input               angle_done1,        // Arrived at target angle
    input               angle_done2,        // Arrived at target angle
    input               angle_done3,        // Arrived at target angle

    output      [7:0]   servo_position0,    // Servo 0 target position
    output      [7:0]   servo_position1,    // Servo 1 target position
    output      [7:0]   servo_position2,    // Servo 2 target position
    output      [7:0]   servo_position3,    // Servo 3 target position

    input       [31:0]  debug_signals,      // Debug signals
    output              led_test_enable,    // Enable the led testing
    output              motor_hot_led,      // Hot motor led
    output              pi_connected_led,   // Orange Pi connected
    output              ps4_connected_led,  // PS4 connected
    output              fault_led           // Fault led
);

// 63:0 is the maximum addressable space with 6-bit address
reg     [7:0]   reg_file    [63:0];
reg     [11:0]  angle_snap0, angle_snap1, angle_snap2, angle_snap3;

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
		reg_file[6'h4]     <=  wr_data[7:0];
end

assign brake0       = reg_file[6'h4][7];
assign enable0      = reg_file[6'h4][6];
assign direction0   = reg_file[6'h4][5];
assign pwm0[4:0]    = reg_file[6'h4][4:0]; 

// ------------- 0x5	DRIVE0_STATUS	-------------
always @(posedge clock) begin
	reg_file[6'h5]     <=  {fault0, startup_fail[0], adc_temp0[5:0]};
end

// ------------- 0x6	DRIVE1_CONTROL	-------------
always @(posedge clock) begin
	if(write_en & ((address == 6'h6) | (address == 6'h3) | (address == 6'h1)))
		reg_file[6'h6]     <=  wr_data[7:0];
end

assign brake1       = reg_file[6'h6][7];
assign enable1      = reg_file[6'h6][6];
assign direction1   = reg_file[6'h6][5];
assign pwm1[4:0]    = reg_file[6'h6][4:0];

// ------------- 0x7	DRIVE1_STATUS	-------------
always @(posedge clock) begin
	reg_file[6'h7]     <=  {fault1, startup_fail[1], adc_temp1[5:0]};
end

// ------------- 0x8	DRIVE2_CONTROL	-------------
always @(posedge clock) begin
	if(write_en & ((address == 6'h8) | (address == 6'h3) | (address == 6'h1)))
		reg_file[6'h8]     <=  wr_data[7:0];
end

assign brake2       = reg_file[6'h8][7];
assign enable2      = reg_file[6'h8][6];
assign direction2   = reg_file[6'h8][5];
assign pwm2[4:0]    = reg_file[6'h8][4:0];

// ------------- 0x9	DRIVE2_STATUS	-------------
always @(posedge clock) begin
	reg_file[6'h9]     <=  {fault2, startup_fail[2], adc_temp2[5:0]};
end

// ------------- 0xA	DRIVE3_CONTROL	-------------
always @(posedge clock) begin
	if(write_en & ((address == 6'hA) | (address == 6'h3) | (address == 6'h1)))
		reg_file[6'hA]     <=  wr_data[7:0];
end

assign brake3       = reg_file[6'hA][7];
assign enable3      = reg_file[6'hA][6];
assign direction3   = reg_file[6'hA][5];
assign pwm3[4:0]    = reg_file[6'hA][4:0];

// ------------- 0xB	DRIVE3_STATUS	-------------
always @(posedge clock) begin
	reg_file[16'hB]     <=  {fault3, startup_fail[3], adc_temp3[5:0]};
end

// ------------- 0xC	ROTATION0_CONTROL	-------------
always @(posedge clock) begin
	if(write_en & (address == 6'hC))
		reg_file[6'hC]     <=  {wr_data[7:5], startup_fail[4], wr_data[3:0]};
end

assign enable4              = reg_file[6'hC][6];
assign enable_stall_chk0    = reg_file[6'hC][5];
assign target_angle0[11:8]  = reg_file[6'hC][3:0];

// ------------- 0xD	ROTATION0_TARGET_ANGLE	-------------
always @(posedge clock) begin
	if(write_en & (address == 6'hD))
		reg_file[6'hD]     <=  wr_data[7:0];
end

assign target_angle0[7:0] = reg_file[6'hD][7:0];

// ------------- 0xE	ROTATION0_CURRENT_ANGLE	-------------
always @(posedge clock) begin
	reg_file[6'hE]     <=  angle_snap0[7:0];
end

// ------------- 0xF	ROTATION0_CURRENT_ANGLE2	-------------
always @(posedge clock) begin
    reg_file[6'hF]     <=  {angle_done0, 3'h0, angle_snap0[11:8]};
    
    if(write_en & (address == 6'hF) & wr_data[4])
        abort_angle0 <= 1'b1;
    else
        abort_angle0 <= 1'b0;
    
    if(write_en & (address == 6'hF) & wr_data[5])
        update_angle0 <= 1'b1;
    else
        update_angle0 <= 1'b0;

    // Grab a snapshot of the angle when bit 6 is written - this ensures
    // that when we do the reading back that we don't read the 1st 8 bits
    // and then the value changes before we read the upper 4 bits
    if(write_en & (address == 6'hF) & wr_data[6]) 
        angle_snap0[11:0] <= current_angle0[11:0];
   
end 

// ------------- 0x10	Kp	-------------
always @(posedge clock) begin
	if(write_en & (address == 6'h10))
		reg_file[6'h10]     <=  wr_data[7:0];
end

assign kp0[7:0]              = reg_file[6'h10][7:0];

// ------------- 0x11	Ki and Kd	-------------
always @(posedge clock) begin
	if(write_en & (address == 6'h11))
		reg_file[6'h11]     <=  wr_data[7:0];
end

assign ki0[3:0]              = reg_file[6'h11][7:4];
assign kd0[3:0]              = reg_file[6'h11][3:0];

// ------------- 0x12	ROTATION_PWM_OVRD	-------------
always @(posedge clock) begin
	if(write_en & (address == 6'h12))
		reg_file[6'h12]     <=  wr_data[7:0];
end

assign rot_pwm_ovrd0         = reg_file[6'h12][7];
assign pwm_dir_ovrd0         = reg_file[6'h12][6];
assign pwm_ratio_ovrd0[5:0]  = reg_file[6'h12][5:0];


// ------------- 0x13	ROTATION1_CONTROL	-------------
always @(posedge clock) begin
	if(write_en & (address == 6'h13))
		reg_file[6'h13]     <=  {wr_data[7:5], startup_fail[4], wr_data[3:0]};
end

assign enable5              = reg_file[6'h13][6];
assign enable_stall_chk1    = reg_file[6'h13][5];
assign target_angle1[11:8]  = reg_file[6'h13][3:0];

// ------------- 0x14	ROTATION1_TARGET_ANGLE	-------------
always @(posedge clock) begin
	if(write_en & (address == 6'h14))
		reg_file[6'h14]     <=  wr_data[7:0];
end

assign target_angle1[7:0] = reg_file[6'h14][7:0];

// ------------- 0x15	ROTATION1_CURRENT_ANGLE	-------------
always @(posedge clock) begin
	reg_file[6'h15]     <=  angle_snap1[7:0];
end

// ------------- 0x16	ROTATION1_CURRENT_ANGLE2	-------------
always @(posedge clock) begin
    reg_file[6'h16]     <=  {angle_done1, 3'h0, angle_snap1[11:8]};
    
    if(write_en & (address == 6'h16) & wr_data[4])
        abort_angle1 <= 1'b1;
    else
        abort_angle1 <= 1'b0;
    
    if(write_en & (address == 6'h16) & wr_data[5])
        update_angle1 <= 1'b1;
    else
        update_angle1 <= 1'b0;

    // Grab a snapshot of the angle when bit 6 is written - this ensures
    // that when we do the reading back that we don't read the 1st 8 bits
    // and then the value changes before we read the upper 4 bits
    if(write_en & (address == 6'h16) & wr_data[6]) 
        angle_snap1[11:0] <= current_angle1[11:0];
   
end 

// ------------- 0x17	Kp	-------------
always @(posedge clock) begin
	if(write_en & (address == 6'h17))
		reg_file[6'h17]     <=  wr_data[7:0];
end

assign kp1[7:0]              = reg_file[6'h17][7:0];

// ------------- 0x18	Ki and Kd	-------------
always @(posedge clock) begin
	if(write_en & (address == 6'h18))
		reg_file[6'h18]     <=  wr_data[7:0];
end

assign ki1[3:0]              = reg_file[6'h18][7:4];
assign kd1[3:0]              = reg_file[6'h18][3:0];

// ------------- 0x19	ROTATION_PWM_TEST	-------------
always @(posedge clock) begin
	if(write_en & (address == 6'h19))
		reg_file[6'h19]     <=  wr_data[7:0];
end

assign rot_pwm_ovrd1         = reg_file[6'h19][7];
assign pwm_dir_ovrd1         = reg_file[6'h19][6];
assign pwm_ratio_ovrd1[5:0]  = reg_file[6'h19][5:0];

// ------------- 0x1A	ROTATION2_CONTROL	------------- 
always @(posedge clock) begin
	if(write_en & (address == 6'h1A))
		reg_file[6'h1A]     <=  {wr_data[7:5], startup_fail[4], wr_data[3:0]};
end

assign enable6              = reg_file[6'h1A][6];
assign enable_stall_chk2    = reg_file[6'h1A][5];
assign target_angle2[11:8]  = reg_file[6'h1A][3:0];

// ------------- 0x1B	ROTATION2_TARGET_ANGLE	-------------
always @(posedge clock) begin
	if(write_en & (address == 6'h1B))
		reg_file[6'h1B]     <=  wr_data[7:0];
end

assign target_angle2[7:0] = reg_file[6'h1B][7:0];

// ------------- 0x1C	ROTATION2_CURRENT_ANGLE	-------------
always @(posedge clock) begin
	reg_file[6'h1C]     <=  angle_snap2[7:0];
end

// ------------- 0x1D	ROTATION2_CURRENT_ANGLE2	-------------
always @(posedge clock) begin
    reg_file[6'h1D]     <=  {angle_done2, 3'h0, angle_snap2[11:8]};
    
    if(write_en & (address == 6'h1D) & wr_data[4])
        abort_angle2 <= 1'b1;
    else
        abort_angle2 <= 1'b0;
    
    if(write_en & (address == 6'h1D) & wr_data[5])
        update_angle2 <= 1'b1;
    else
        update_angle2 <= 1'b0;

    // Grab a snapshot of the angle when bit 6 is written - this ensures
    // that when we do the reading back that we don't read the 1st 8 bits
    // and then the value changes before we read the upper 4 bits
    if(write_en & (address == 6'h1D) & wr_data[6]) 
        angle_snap2[11:0] <= current_angle2[11:0];
   
end 

// ------------- 0x1E	Kp	-------------
always @(posedge clock) begin
	if(write_en & (address == 6'h1E))
		reg_file[6'h1E]     <=  wr_data[7:0];
end

assign kp2[7:0]              = reg_file[6'h1E][7:0];

// ------------- 0x1F	Ki and Kd	-------------
always @(posedge clock) begin
	if(write_en & (address == 6'h1F))
		reg_file[6'h1F]     <=  wr_data[7:0];
end

assign ki2[3:0]              = reg_file[6'h1F][7:4];
assign kd2[3:0]              = reg_file[6'h1F][3:0];

// ------------- 0x20	ROTATION_PWM_TEST	-------------
always @(posedge clock) begin
	if(write_en & (address == 6'h20))
		reg_file[6'h20]     <=  wr_data[7:0];
end

assign rot_pwm_ovrd2         = reg_file[6'h20][7];
assign pwm_dir_ovrd2         = reg_file[6'h20][6];
assign pwm_ratio_ovrd2[5:0]  = reg_file[6'h20][5:0];

// ------------- 0x21	ROTATION3_CONTROL	------------- 
always @(posedge clock) begin
	if(write_en & (address == 6'h21))
		reg_file[6'h21]     <=  {wr_data[7:5], startup_fail[4], wr_data[3:0]};
end

assign enable7              = reg_file[6'h21][6];
assign enable_stall_chk3    = reg_file[6'h21][5];
assign target_angle3[11:8]  = reg_file[6'h21][3:0];

// ------------- 0x22	ROTATION3_TARGET_ANGLE	-------------
always @(posedge clock) begin
	if(write_en & (address == 6'h22))
		reg_file[6'h22]     <=  wr_data[7:0];
end

assign target_angle3[7:0] = reg_file[6'h22][7:0];

// ------------- 0x23	ROTATION2_CURRENT_ANGLE	-------------
always @(posedge clock) begin
	reg_file[6'h23]     <=  angle_snap3[7:0];
end

// ------------- 0x24	ROTATION2_CURRENT_ANGLE2	------------- 
always @(posedge clock) begin
    reg_file[6'h24]     <=  {angle_done3, 3'h0, angle_snap3[11:8]};
    
    if(write_en & (address == 6'h24) & wr_data[4])
        abort_angle3 <= 1'b1;
    else
        abort_angle3 <= 1'b0;
    
    if(write_en & (address == 6'h24) & wr_data[5])
        update_angle3 <= 1'b1;
    else
        update_angle3 <= 1'b0;

    // Grab a snapshot of the angle when bit 6 is written - this ensures
    // that when we do the reading back that we don't read the 1st 8 bits
    // and then the value changes before we read the upper 4 bits
    if(write_en & (address == 6'h24) & wr_data[6]) 
        angle_snap3[11:0] <= current_angle3[11:0];
   
end 

// ------------- 0x25	Kp	-------------
always @(posedge clock) begin
	if(write_en & (address == 6'h25))
		reg_file[6'h25]     <=  wr_data[7:0];
end

assign kp3[7:0]              = reg_file[6'h25][7:0];

// ------------- 0x26	Ki and Kd	-------------
always @(posedge clock) begin
	if(write_en & (address == 6'h26))
		reg_file[6'h26]     <=  wr_data[7:0];
end

assign ki3[3:0]              = reg_file[6'h26][7:4];
assign kd3[3:0]              = reg_file[6'h26][3:0];

// ------------- 0x27	ROTATION_PWM_TEST	-------------
always @(posedge clock) begin
	if(write_en & (address == 6'h27))
		reg_file[6'h27]     <=  wr_data[7:0];
end

assign rot_pwm_ovrd3         = reg_file[6'h27][7];
assign pwm_dir_ovrd3         = reg_file[6'h27][6];
assign pwm_ratio_ovrd3[5:0]  = reg_file[6'h27][5:0];

// --------------- 0x30	SERVO0_CONTROL	----------------
always @(posedge clock) begin
	if(write_en & (address == 6'h30))
		reg_file[6'h30]     <=  wr_data[7:0];
end

assign servo_position0[7:0]    = reg_file[6'h30][7:0];

// --------------- 0x31	SERVO1_CONTROL	----------------
always @(posedge clock) begin
	if(write_en & (address == 6'h31))
		reg_file[6'h31]     <=  wr_data[7:0];
end

assign servo_position1[7:0]    = reg_file[6'h31][7:0];

// --------------- 0x32	SERVO2_CONTROL	----------------
always @(posedge clock) begin
	if(write_en & (address == 6'h32))
		reg_file[6'h32]     <=  wr_data[7:0];
end

assign servo_position2[7:0]    = reg_file[6'h32][7:0];

// --------------- 0x33	SERVO3_CONTROL	----------------
always @(posedge clock) begin
	if(write_en & (address == 6'h33))
		reg_file[6'h33]     <=  wr_data[7:0];
end

assign servo_position3[7:0]    = reg_file[6'h33][7:0];

// ---------------   0x34	DEBUG   ----------------
always @(posedge clock) begin
	reg_file[6'h34]     <=  debug_signals[7:0];
end

// ---------------   0x35	DEBUG   ----------------
always @(posedge clock) begin
	reg_file[6'h35]     <=  debug_signals[15:8];
end

// ---------------   0x36	DEBUG   ----------------
always @(posedge clock) begin
	reg_file[6'h36]     <=  debug_signals[23:16];
end

// ---------------   0x37	DEBUG   ----------------
always @(posedge clock) begin
	reg_file[6'h37]     <=  debug_signals[31:24];
end

// ------------- 0x38	LED_TEST	-------------
always @(posedge clock) begin
	if(write_en & (address == 6'h38))
		reg_file[6'h38]     <=  wr_data[7:0];
end

assign led_test_enable      = reg_file[6'h38][4];
assign motor_hot_led        = reg_file[6'h38][3];
assign ps4_connected_led    = reg_file[6'h38][2];
assign pi_connected_led     = reg_file[6'h38][1];
assign fault_led            = reg_file[6'h38][0];

endmodule
