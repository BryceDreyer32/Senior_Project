// Copyright 2024
// Bryce's Senior Project
// Description: This is the RTL for Address Decoder and Register File

`define DRIVE0_CONTROL				6'h4
`define DRIVE1_CONTROL				6'h5
`define DRIVE2_CONTROL				6'h6
`define DRIVE3_CONTROL				6'h7
`define ROTATION0_CONTROL			6'h8
`define ROTATION0_CURRENT_ANGLE		6'h9
`define ROTATION0_CURRENT_ANGLE2	6'hA
`define ROTATION1_CONTROL			6'hB
`define ROTATION1_CURRENT_ANGLE		6'hC
`define ROTATION1_CURRENT_ANGLE2	6'hD
`define ROTATION2_CONTROL			6'hE
`define ROTATION2_CURRENT_ANGLE		6'hF
`define ROTATION2_CURRENT_ANGLE2	6'h10
`define ROTATION3_CONTROL			6'h11
`define ROTATION3_CURRENT_ANGLE		6'h12
`define ROTATION3_CURRENT_ANGLE2	6'h13
`define SERVO0_CONTROL				6'h14
`define SERVO1_CONTROL				6'h15
`define SERVO2_CONTROL				6'h16
`define SERVO3_CONTROL				6'h17
`define DEBUG0						6'h18
`define DEBUG1						6'h19
`define DEBUG2						6'h1A
`define LED_TEST					6'h1B

module reg_file ( 
    input           reset_n,   	     // Active low reset
    input           clock,     	     // The main clock
    input   [5:0]   address,   	     // Read / write address
    input           write_en,  	     // Write enable
    input   [7:0]   wr_data,   	     // Write data
    input           read_en,  	     // Read enable
    output reg [7:0]   rd_data,   	 // Read data
								     
    // DRIVE MOTORS
    output              drv_pwm_enable0,    // Drive motor enable
    output              drv_pwm_dir0,       // Drive motor direction
    output      [5:0]   drv_pwm_ratio0,     // Drive motor value

    output              drv_pwm_enable1,    // Drive motor enable
    output              drv_pwm_dir1,       // Drive motor direction
    output      [5:0]   drv_pwm_ratio1,     // Drive motor value

    output              drv_pwm_enable2,    // Drive motor enable
    output              drv_pwm_dir2,       // Drive motor direction
    output      [5:0]   drv_pwm_ratio2,     // Drive motor value

    output              drv_pwm_enable3,    // Drive motor enable
    output              drv_pwm_dir3,       // Drive motor direction
    output      [5:0]   drv_pwm_ratio3,     // Drive motor value


    // ROTATION MOTORS
    output              rot_pwm_enable0,    // Rotation motor enable
    output              rot_pwm_dir0,       // Rotation motor direction
    output      [5:0]   rot_pwm_ratio0,     // Rotation motor value

    output              rot_pwm_enable1,    // Rotation motor enable
    output              rot_pwm_dir1,       // Rotation motor direction
    output      [5:0]   rot_pwm_ratio1,     // Rotation motor value

    output              rot_pwm_enable2,    // Rotation motor enable
    output              rot_pwm_dir2,       // Rotation motor direction
    output      [5:0]   rot_pwm_ratio2,     // Rotation motor value

    output              rot_pwm_enable3,    // Rotation motor enable
    output              rot_pwm_dir3,       // Rotation motor direction
    output      [5:0]   rot_pwm_ratio3,     // Rotation motor value

    // ENCODER VALUES
    input       [11:0]  current_angle0,     // AS5600 Raw Angle
    input       [11:0]  current_angle1,     // AS5600 Raw Angle
    input       [11:0]  current_angle2,     // AS5600 Raw Angle
    input       [11:0]  current_angle3,     // AS5600 Raw Angle

    output      [7:0]   servo_position0,    // Servo 0 target position
    output      [7:0]   servo_position1,    // Servo 1 target position
    output      [7:0]   servo_position2,    // Servo 2 target position
    output      [7:0]   servo_position3,    // Servo 3 target position

    input       [31:0]  debug_signals,      // Debug signals
    output      [3:0]   led_pwm,            // LED intesity
    output              led_test_enable,    // Enable the led testing
    output              motor_hot_led,      // Hot motor led
    output              pi_connected_led,   // Orange Pi connected
    output              ps4_connected_led,  // PS4 connected
    output              fault_led           // Fault led
);

// 63:0 is the maximum addressable space with 6-bit address
reg     [7:0]   reg_file    [26:0];
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

// ------------- 	DRIVE0_CONTROL	-------------
always @(posedge clock) begin
	if(write_en & ((address == `DRIVE0_CONTROL) | (address == 6'h3) | (address == 6'h1)))
		reg_file[`DRIVE0_CONTROL]     <=  wr_data[7:0];
end

assign drv_pwm_enable0  = reg_file[`DRIVE0_CONTROL][7];
assign drv_pwm_dir0     = reg_file[`DRIVE0_CONTROL][6];
assign drv_pwm_ratio0   = reg_file[`DRIVE0_CONTROL][5:0]; 

// ------------- 	DRIVE1_CONTROL	-------------
always @(posedge clock) begin
	if(write_en & ((address == `DRIVE1_CONTROL) | (address == 6'h3) | (address == 6'h1)))
		reg_file[`DRIVE1_CONTROL]     <=  wr_data[7:0];
end

assign drv_pwm_enable1  = reg_file[`DRIVE1_CONTROL][7];
assign drv_pwm_dir1     = reg_file[`DRIVE1_CONTROL][6];
assign drv_pwm_ratio1   = reg_file[`DRIVE1_CONTROL][5:0]; 

// ------------- 	DRIVE2_CONTROL	-------------
always @(posedge clock) begin
	if(write_en & ((address == `DRIVE2_CONTROL) | (address == 6'h3) | (address == 6'h1)))
		reg_file[`DRIVE2_CONTROL]     <=  wr_data[7:0];
end

assign drv_pwm_enable2  = reg_file[`DRIVE2_CONTROL][7];
assign drv_pwm_dir2     = reg_file[`DRIVE2_CONTROL][6];
assign drv_pwm_ratio2   = reg_file[`DRIVE2_CONTROL][5:0]; 

// ------------- 	DRIVE3_CONTROL	-------------
always @(posedge clock) begin
	if(write_en & ((address == `DRIVE3_CONTROL) | (address == 6'h3) | (address == 6'h1)))
		reg_file[`DRIVE3_CONTROL]     <=  wr_data[7:0];
end

assign drv_pwm_enable3  = reg_file[`DRIVE3_CONTROL][7];
assign drv_pwm_dir3     = reg_file[`DRIVE3_CONTROL][6];
assign drv_pwm_ratio3   = reg_file[`DRIVE3_CONTROL][5:0]; 

// ------------- 	ROTATION0_CONTROL	-------------
always @(posedge clock) begin
	if(write_en & (address == `ROTATION0_CONTROL))
		reg_file[`ROTATION0_CONTROL]     <=  wr_data[7:0];
end

assign rot_pwm_enable0      = reg_file[`ROTATION0_CONTROL][7];
assign rot_pwm_dir0         = reg_file[`ROTATION0_CONTROL][6];
assign rot_pwm_ratio0       = reg_file[`ROTATION0_CONTROL][5:0];

// ------------- 	ROTATION0_CURRENT_ANGLE	-------------
always @(posedge clock) begin
	reg_file[`ROTATION0_CURRENT_ANGLE]     <=  angle_snap0[7:0];
end

// ------------- 	ROTATION0_CURRENT_ANGLE2	-------------
always @(posedge clock) begin
    reg_file[`ROTATION0_CURRENT_ANGLE2]     <=  {4'h0, angle_snap0[11:8]};
    
    // Grab a snapshot of the angle when bit 6 is written - this ensures
    // that when we do the reading back that we don't read the 1st 8 bits
    // and then the value changes before we read the upper 4 bits
    if(write_en & (address == `ROTATION0_CURRENT_ANGLE2) & wr_data[6]) 
        angle_snap0[11:0] <= current_angle0[11:0];   
end 



// ------------- 	ROTATION1_CONTROL	-------------
always @(posedge clock) begin
	if(write_en & (address == `ROTATION1_CONTROL))
		reg_file[`ROTATION1_CONTROL]     <=  wr_data[7:0];
end

assign rot_pwm_enable1      = reg_file[`ROTATION1_CONTROL][7];
assign rot_pwm_dir1         = reg_file[`ROTATION1_CONTROL][6];
assign rot_pwm_ratio1       = reg_file[`ROTATION1_CONTROL][5:0];

// ------------- 	ROTATION1_CURRENT_ANGLE	-------------
always @(posedge clock) begin
	reg_file[`ROTATION1_CURRENT_ANGLE]     <=  angle_snap1[7:0];
end

// ------------- 	ROTATION1_CURRENT_ANGLE2	-------------
always @(posedge clock) begin
    reg_file[`ROTATION1_CURRENT_ANGLE2]     <=  {4'h0, angle_snap1[11:8]};
    
    // Grab a snapshot of the angle when bit 6 is written - this ensures
    // that when we do the reading back that we don't read the 1st 8 bits
    // and then the value changes before we read the upper 4 bits
    if(write_en & (address == `ROTATION1_CURRENT_ANGLE2) & wr_data[6]) 
        angle_snap1[11:0] <= current_angle1[11:0];   
end


// ------------- 	ROTATION2_CONTROL	-------------
always @(posedge clock) begin
	if(write_en & (address == `ROTATION2_CONTROL))
		reg_file[`ROTATION2_CONTROL]     <=  wr_data[7:0];
end

assign rot_pwm_enable2      = reg_file[`ROTATION2_CONTROL][7];
assign rot_pwm_dir2         = reg_file[`ROTATION2_CONTROL][6];
assign rot_pwm_ratio2       = reg_file[`ROTATION2_CONTROL][5:0];

// ------------- 	ROTATION2_CURRENT_ANGLE	-------------
always @(posedge clock) begin
	reg_file[`ROTATION2_CURRENT_ANGLE]     <=  angle_snap2[7:0];
end

// ------------- 	ROTATION2_CURRENT_ANGLE2	-------------
always @(posedge clock) begin
    reg_file[`ROTATION2_CURRENT_ANGLE2]     <=  {4'h0, angle_snap2[11:8]};
    
    // Grab a snapshot of the angle when bit 6 is written - this ensures
    // that when we do the reading back that we don't read the 1st 8 bits
    // and then the value changes before we read the upper 4 bits
    if(write_en & (address == `ROTATION2_CURRENT_ANGLE2) & wr_data[6]) 
        angle_snap2[11:0] <= current_angle2[11:0];   
end



// ------------- 	ROTATION3_CONTROL	-------------
always @(posedge clock) begin
	if(write_en & (address == `ROTATION3_CONTROL))
		reg_file[`ROTATION3_CONTROL]     <=  wr_data[7:0];
end

assign rot_pwm_enable3      = reg_file[`ROTATION3_CONTROL][7];
assign rot_pwm_dir3         = reg_file[`ROTATION3_CONTROL][6];
assign rot_pwm_ratio3       = reg_file[`ROTATION3_CONTROL][5:0];

// ------------- 	ROTATION3_CURRENT_ANGLE	-------------
always @(posedge clock) begin
	reg_file[`ROTATION3_CURRENT_ANGLE]     <=  angle_snap3[7:0];
end

// ------------- 	ROTATION3_CURRENT_ANGLE2	-------------
always @(posedge clock) begin
    reg_file[`ROTATION3_CURRENT_ANGLE2]     <=  {4'h0, angle_snap3[11:8]};
    
    // Grab a snapshot of the angle when bit 6 is written - this ensures
    // that when we do the reading back that we don't read the 1st 8 bits
    // and then the value changes before we read the upper 4 bits
    if(write_en & (address == `ROTATION3_CURRENT_ANGLE2) & wr_data[6]) 
        angle_snap3[11:0] <= current_angle3[11:0];   
end

// --------------- 	SERVO0_CONTROL	----------------
always @(posedge clock) begin
	if(write_en & (address == `SERVO0_CONTROL))
		reg_file[`SERVO0_CONTROL]     <=  wr_data[7:0];
end

assign servo_position0[7:0]    = reg_file[`SERVO0_CONTROL][7:0];

// --------------- 	SERVO1_CONTROL	----------------
always @(posedge clock) begin
	if(write_en & (address == `SERVO1_CONTROL))
		reg_file[`SERVO1_CONTROL]     <=  wr_data[7:0];
end

assign servo_position1[7:0]    = reg_file[`SERVO1_CONTROL][7:0];

// --------------- 	SERVO2_CONTROL	----------------
always @(posedge clock) begin
	if(write_en & (address == `SERVO2_CONTROL))
		reg_file[`SERVO2_CONTROL]     <=  wr_data[7:0];
end

assign servo_position2[7:0]    = reg_file[`SERVO2_CONTROL][7:0];

// --------------- 	SERVO3_CONTROL	----------------
always @(posedge clock) begin
	if(write_en & (address == `SERVO3_CONTROL))
		reg_file[`SERVO3_CONTROL]     <=  wr_data[7:0];
end

assign servo_position3[7:0]    = reg_file[`SERVO3_CONTROL][7:0];

// ---------------   	DEBUG   ----------------
always @(posedge clock) begin
	reg_file[`DEBUG0]     <=  debug_signals[7:0];
end

// ---------------   	DEBUG   ----------------
always @(posedge clock) begin
	reg_file[`DEBUG1]     <=  debug_signals[15:8];
end

// ---------------   	DEBUG   ----------------
always @(posedge clock) begin
	reg_file[`DEBUG2]     <=  debug_signals[23:16];
end

// ------------- 	LED_TEST	-------------
always @(posedge clock) begin
	if(write_en & (address == `LED_TEST))
		reg_file[`LED_TEST]     <=  wr_data[7:0];
end

assign led_pwm              = reg_file[`LED_TEST][7:4];
assign led_test_enable      = reg_file[`LED_TEST][3];
assign ps4_connected_led    = reg_file[`LED_TEST][2];
assign pi_connected_led     = reg_file[`LED_TEST][1];
assign fault_led            = reg_file[`LED_TEST][0];

endmodule
