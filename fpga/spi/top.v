// Copyright 2024
// Bryce's Senior Project
// Description: This is the RTL for the SPI top-level for test purposes
module top(
    input                   clock,      // The main clock
    input                   spi_clk,    // The SPI clock
    input                   cs_n,       // Active low chip select
    input                   mosi,       // Master out slave in
    output                  miso,        // Master in slave out (SPI mode 0)    
    
    // Only for debug use, leave out on real design
    output  wire            write_en,
    output  wire            read_en
);

    wire            reset_n;
    wire   [5:0]    address;   	     // Read / write address
//    wire            write_en;  	     // Write enable
    wire   [7:0]    wr_data;   	     // Write data
//    wire            read_en;  	     // Read enable
    wire  [7:0]     rd_data;   	     // Read data

    wire          fault0;    	     // Fault signal from motor
    wire  [6:0]   adc_temp0; 	     // Adc temperature from motor
    wire          fault1;    	     // Fault signal from motor
    wire  [6:0]   adc_temp1; 	     // Adc temperature from motor
    wire          fault2;    	     // Fault signal from motor
    wire  [6:0]   adc_temp2; 	     // Adc temperature from motor
    wire          fault3;    	     // Fault signal from motor
    wire  [6:0]   adc_temp3; 	     // Adc temperature from motor
    wire          fault4;    	     // Fault signal from motor
    wire  [6:0]   adc_temp4; 	     // Adc temperature from motor
    wire          fault5;    	     // Fault signal from motor
    wire  [6:0]   adc_temp5; 	     // Adc temperature from motor
    wire          fault6;    	     // Fault signal from motor
    wire  [6:0]   adc_temp6; 	     // Adc temperature from motor
    wire          fault7;    	     // Fault signal from motor
    wire  [6:0]   adc_temp7; 	     // Adc temperature from motor

    wire          brake0;    	     // Brake control
    wire          enable0;   	     // Motor enable
    wire          direction0;	     // Motor direction
    wire  [4:0]   pwm0;      	     // PWM control  
    wire          brake1;    	     // Brake control
    wire          enable1;   	     // Motor enable
    wire          direction1;	     // Motor direction
    wire  [4:0]   pwm1;      	     // PWM control  
    wire          brake2;    	     // Brake control
    wire          enable2;   	     // Motor enable
    wire          direction2;	     // Motor direction
    wire  [4:0]   pwm2;      	     // PWM control  
    wire          brake3;    	     // Brake control
    wire          enable3;   	     // Motor enable
    wire          direction3;	     // Motor direction
    wire  [4:0]   pwm3;      	     // PWM control
    wire          brake4;    	     // Brake control
    wire          enable4;   	     // Motor enable
    wire          direction4;	     // Motor direction
    wire  [4:0]   pwm4;      	     // PWM control  
    wire          brake5;    	     // Brake control
    wire          enable5;   	     // Motor enable
    wire          direction5;	     // Motor direction
    wire  [4:0]   pwm5;      	     // PWM control  
    wire          brake6;    	     // Brake control
    wire          enable6;   	     // Motor enable
    wire          direction6;	     // Motor direction
    wire  [4:0]   pwm6;      	     // PWM control  
    wire          brake7;    	     // Brake control
    wire          enable7;   	     // Motor enable
    wire          direction7;	     // Motor direction
    wire  [4:0]   pwm7;      	     // PWM control  

    wire  [11:0]  target_angle0;   // Rotation target angle
    wire  [11:0]  current_angle0;  // The current angle
    wire  [11:0]  target_angle1;   // Rotation target angle
    wire  [11:0]  current_angle1;  // The current angle
    wire  [11:0]  target_angle2;   // Rotation target angle
    wire  [11:0]  current_angle2;  // The current angle
    wire  [11:0]  target_angle3;   // Rotation target angle
    wire  [11:0]  current_angle3;  // The current angle

    wire  [7:0]   servo_position0; // Servo 0 target position
    wire  [7:0]   servo_position1; // Servo 1 target position
    wire  [7:0]   servo_position2; // Servo 2 target position
    wire  [7:0]   servo_position3; // Servo 3 target position


spi spi(
	.reset_n            (reset_n),          // Active low reset
	.clock              (clock),            // The main clock
	.spi_clk            (spi_clk),          // The SPI clock
	.cs_n               (cs_n),             // Active low chip select
	.mosi               (mosi),             // Master out slave in
	.miso               (miso),             // Master in slave out (SPI mode 0)
    .address            (address),   	    // Read / write address
    .write_en           (write_en),  	    // Write enable
    .wr_data            (wr_data),   	    // Write data
    .read_en            (read_en),   	    // Read enable
    .rd_data            (rd_data)   	    // Read data);
);

reg_file rf(
    .reset_n            (reset_n),   	    // Active low reset
    .clock              (clock),     	    // The main clock
    .address            (address),   	    // Read / write address
    .write_en           (write_en),  	    // Write enable
    .wr_data            (wr_data),   	    // Write data
    .read_en            (read_en),   	    // Read enable
    .rd_data            (rd_data),   	    // Read data
				     
    .fault0             (fault0),    	    // Fault signal from motor
    .adc_temp0          (adc_temp0), 	    // Adc temperature from motor
    .fault1             (fault1),    	    // Fault signal from motor
    .adc_temp1          (adc_temp1), 	    // Adc temperature from motor
    .fault2             (fault2),    	    // Fault signal from motor
    .adc_temp2          (adc_temp2), 	    // Adc temperature from motor
    .fault3             (fault3),    	    // Fault signal from motor
    .adc_temp3          (adc_temp3), 	    // Adc temperature from motor
    .fault4             (fault4),    	    // Fault signal from motor
    .adc_temp4          (adc_temp4), 	    // Adc temperature from motor
    .fault5             (fault5),    	    // Fault signal from motor
    .adc_temp5          (adc_temp5), 	    // Adc temperature from motor
    .fault6             (fault6),    	    // Fault signal from motor
    .adc_temp6          (adc_temp6), 	    // Adc temperature from motor
    .fault7             (fault7),    	    // Fault signal from motor
    .adc_temp7          (adc_temp7), 	    // Adc temperature from motor
				     
	.brake0             (/*brake0*/),    	    // Brake control
    .enable0            (/*enable0*/),   	    // Motor enable
    .direction0         (/*direction0*/),	    // Motor direction
    .pwm0               (/*pwm0*/),      	    // PWM control  
    .brake1             (/*brake1*/),    	    // Brake control
    .enable1            (/*enable1*/),   	    // Motor enable
    .direction1         (/*direction1*/),	    // Motor direction
    .pwm1               (/*pwm1*/),      	    // PWM control  
    .brake2             (/*brake2*/),    	    // Brake control
    .enable2            (/*enable2*/),   	    // Motor enable
    .direction2         (/*direction2*/),	    // Motor direction
    .pwm2               (/*pwm2*/),      	    // PWM control  
    .brake3             (/*brake3*/),    	    // Brake control
    .enable3            (/*enable3*/),   	    // Motor enable
    .direction3         (/*direction3*/),	    // Motor direction
    .pwm3               (/*pwm3*/),      	    // PWM control
    .brake4             (/*brake4*/),    	    // Brake control
    .enable4            (/*enable4*/),   	    // Motor enable
    .direction4         (/*direction4*/),	    // Motor direction
    .brake5             (/*brake5*/),    	    // Brake control
    .enable5            (/*enable5*/),   	    // Motor enable
    .direction5         (/*direction5*/),	    // Motor direction
    .brake6             (/*brake6*/),    	    // Brake control
    .enable6            (/*enable6*/),   	    // Motor enable
    .direction6         (/*direction6*/),	    // Motor direction 
    .brake7             (/*brake7*/),    	    // Brake control
    .enable7            (/*enable7*/),   	    // Motor enable
    .direction7         (/*direction7*/),	    // Motor direction
					 
    .target_angle0      (target_angle0),    // Rotation target angle
    .current_angle0     (current_angle0),   // The current angle
    .target_angle1      (target_angle1),    // Rotation target angle
    .current_angle1     (current_angle1),   // The current angle
    .target_angle2      (target_angle2),    // Rotation target angle
    .current_angle2     (current_angle2),   // The current angle
    .target_angle3      (target_angle3),    // Rotation target angle
    .current_angle3     (current_angle3),   // The current angle
	
    .servo_position0    (servo_position0), // Servo 0 target position
    .servo_position1    (servo_position1), // Servo 1 target position
    .servo_position2    (servo_position2), // Servo 2 target position
    .servo_position3    (servo_position3)  // Servo 3 target position
);

assign reset_n = 1;
assign fault0 = 0;
assign fault1 = 0;
assign fault2 = 0;
assign fault3 = 0;
assign adc_temp0 = 'hA;
assign adc_temp1 = 'hA;
assign adc_temp2 = 'hA;
assign adc_temp3 = 'hA;
assign current_angle0 = 'h55;
assign current_angle1 = 'h55;
assign current_angle2 = 'h55;
assign current_angle3 = 'h55;

endmodule