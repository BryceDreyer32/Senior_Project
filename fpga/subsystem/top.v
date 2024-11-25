// Copyright 2024
// Bryce's Senior Project
// Description: This is the RTL for the FPGA Subsystem
module top(
    // Clock and Reset
    //input           reset_n,        // Active low reset
    input           clock,          // The main clock
    
    // SPI Interface
    input           spi_clock,      // The SPI clock
    input           cs_n,           // Active low chip select
    input           mosi,           // Master out slave in (SPI mode 0)
    output          miso,           // Master in slave out (SPI mode 0)

    // Swerve Rotation Motors
    output  [3:0]   sr_pwm,         // The swerve rotation PWM wave
    input   [3:0]   sr_fault,       // Swerve rotation fault from motor controller
    output  [3:0]   sr_direction,   // Swerve rotation direction to motor controller
    output  [3:0]   sr_enable,      // Swerve rotation enable to motor controller
    output  [3:0]   sr_brake,       // Swerve rotation brake to motor controller
    output  [3:0]   scl,            // The I2C clock to encoders
    inout   [3:0]   sda,            // The I2C bi-directional data to/from encoders

    // Swerve Drive Motors
    output  [3:0]   sd_uart,        // The swerve drive UART

    // Arm Servos
    output  wire [3:0]   servo_pwm,      // The arm servo PWM wave
    
    // Status and Config
    input           tang_config,    // A 1-bit pull high or low for general configuration
    output          status_fault,   // Control for LED for when a fault has occurred
    output          status_pi,      // Control for LED for when the Orange Pi is connected
    output          status_ps4,     // Control for LED for when the PS4 controller is connected
    output          status_debug    // Control for LED for general debug
);

    assign          servo_pwm[0] = write_en;
    assign          servo_pwm[1] = read_en;

    wire            reset_n = 1'b1;  // reset

    wire  [5:0]     address;   	     // Read / write address
    wire            write_en;  	     // Write enable
    wire  [7:0]     wr_data;   	     // Write data
    wire            read_en;  	     // Read enable
    wire  [7:0]     rd_data;   	     // Read data

    wire            fault0;    	     // Fault signal from motor
    wire  [6:0]     adc_temp0; 	     // Adc temperature from motor
    wire            fault1;    	     // Fault signal from motor
    wire  [6:0]     adc_temp1; 	     // Adc temperature from motor
    wire            fault2;    	     // Fault signal from motor
    wire  [6:0]     adc_temp2; 	     // Adc temperature from motor
    wire            fault3;    	     // Fault signal from motor
    wire  [6:0]     adc_temp3; 	     // Adc temperature from motor
    wire            fault4;    	     // Fault signal from motor
    wire  [6:0]     adc_temp4; 	     // Adc temperature from motor
    wire            fault5;    	     // Fault signal from motor
    wire  [6:0]     adc_temp5; 	     // Adc temperature from motor
    wire            fault6;    	     // Fault signal from motor
    wire  [6:0]     adc_temp6; 	     // Adc temperature from motor
    wire            fault7;    	     // Fault signal from motor
    wire  [6:0]     adc_temp7; 	     // Adc temperature from motor

    wire            brake0;    	     // Brake control
    wire            enable0;   	     // Motor enable
    wire            direction0;	     // Motor direction
    wire  [4:0]     pwm0;      	     // PWM control  
    wire            brake1;    	     // Brake control
    wire            enable1;   	     // Motor enable
    wire            direction1;	     // Motor direction
    wire  [4:0]     pwm1;      	     // PWM control  
    wire            brake2;    	     // Brake control
    wire            enable2;   	     // Motor enable
    wire            direction2;	     // Motor direction
    wire  [4:0]     pwm2;      	     // PWM control  
    wire            brake3;    	     // Brake control
    wire            enable3;   	     // Motor enable
    wire            direction3;	     // Motor direction
    wire  [4:0]     pwm3;      	     // PWM control
    wire            brake4;    	     // Brake control
    wire            enable4;   	     // Motor enable
    wire            direction4;	     // Motor direction
    wire  [4:0]     pwm4;      	     // PWM control  
    wire            brake5;    	     // Brake control
    wire            enable5;   	     // Motor enable
    wire            direction5;	     // Motor direction
    wire  [4:0]     pwm5;      	     // PWM control  
    wire            brake6;    	     // Brake control
    wire            enable6;   	     // Motor enable
    wire            direction6;	     // Motor direction
    wire  [4:0]     pwm6;      	     // PWM control  
    wire            brake7;    	     // Brake control
    wire            enable7;   	     // Motor enable
    wire            direction7;	     // Motor direction
    wire  [4:0]     pwm7;      	     // PWM control  

    wire  [7:0]     target_angle0;   // Rotation target angle
    wire  [7:0]     current_angle0;  // The current angle
    wire  [7:0]     target_angle1;   // Rotation target angle
    wire  [7:0]     current_angle1;  // The current angle
    wire  [7:0]     target_angle2;   // Rotation target angle
    wire  [7:0]     current_angle2;  // The current angle
    wire  [7:0]     target_angle3;   // Rotation target angle
    wire  [7:0]     current_angle3;  // The current angle

    wire  [7:0]     servo_position0; // Servo 0 target position
    wire  [7:0]     servo_position1; // Servo 1 target position
    wire  [7:0]     servo_position2; // Servo 2 target position
    wire  [7:0]     servo_position3; // Servo 3 target position


    wire  [7:0]     BAUD_DIVISION = 8'd116;   // Select baud 115200

////////////////////////////////////////////////////////////////
// SPI Controller
////////////////////////////////////////////////////////////////
spi spi(
	.reset_n            (reset_n),          // Active low reset
	.clock              (clock),            // The main clock
	.spi_clk            (spi_clock),        // The SPI clock
	.cs_n               (cs_n),             // Active low chip select
	.mosi               (mosi),             // Master out slave in
	.miso               (miso),             // Master in slave out (SPI mode 0)
    .address            (address[5:0]),	    // Read / write address
    .write_en           (write_en),  	    // Write enable
    .wr_data            (wr_data[7:0]),	    // Write data
    .read_en            (read_en),   	    // Read enable
    .rd_data            (rd_data[7:0]) 	    // Read data
);

////////////////////////////////////////////////////////////////
// Register File (with address decode)
////////////////////////////////////////////////////////////////
reg_file rf(
    .reset_n            (reset_n),   	    // Active low reset
    .clock              (clock),     	    // The main clock
    .address            (address[5:0]),	    // Read / write address
    .write_en           (write_en),  	    // Write enable
    .wr_data            (wr_data[7:0]),	    // Write data
    .read_en            (read_en),   	    // Read enable
    .rd_data            (rd_data[7:0]),	    // Read data
				     
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
				     
	.brake0             (brake0),    	    // Brake control
    .enable0            (enable0),   	    // Motor enable
    .direction0         (direction0),	    // Motor direction
    .pwm0               (pwm0),      	    // PWM control  
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
    .pwm4               (/*pwm4*/),      	    // PWM control  
    .brake5             (/*brake5*/),    	    // Brake control
    .enable5            (/*enable5*/),   	    // Motor enable
    .direction5         (/*direction5*/),	    // Motor direction
    .pwm5               (/*pwm5*/),      	    // PWM control  
    .brake6             (/*brake6*/),    	    // Brake control
    .enable6            (/*enable6*/),   	    // Motor enable
    .direction6         (/*direction6*/),	    // Motor direction
    .pwm6               (/*pwm6*/),      	    // PWM control  
    .brake7             (/*brake7*/),    	    // Brake control
    .enable7            (/*enable7*/),   	    // Motor enable
    .direction7         (/*direction7*/),	    // Motor direction
    .pwm7               (/*pwm7*/),      	    // PWM control  
					 
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

////////////////////////////////////////////////////////////////
// UART Drive Motor0
////////////////////////////////////////////////////////////////
uart uart_drv_0(
    .reset_n            (reset_n),              // Active low reset
    .clock              (clock),                // The main clock
    .tx_data            ({  brake0, 
                            enable0,
                            direction0,
                            pwm0[4:0]}),        // This the 8-bits of data
    .baud_division      (BAUD_DIVISION[7:0]),   // The division ratio to achieve the desired baud rate
    .tx_start           (1'b1),                 // Signal to indicate that the transmission needs to start
    .uart_tx            (sd_uart[0])            // UART_TX
);


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

/*
////////////////////////////////////////////////////////////////
// Address Decoder
////////////////////////////////////////////////////////////////
addr_dec ad0(	
    .reset_n			    (reset_n),
	.spi_clock              (spi_clock),
	.fpga_clock             (clock),
	.cs_n                   (cs_n),   
	.spi_out                (spi_out[15:0]),
	.pwm_done               (pwm_done),
	.sr0_pwm_target         (sr_pwm_target[0]),
	.sr1_pwm_target         (sr_pwm_target[1]),
	.sr2_pwm_target         (sr_pwm_target[2]),
	.sr3_pwm_target         (sr_pwm_target[3]),
	.sd0_pwm_target         (sd_pwm_ratio[0]),
	.sd1_pwm_target         (sd_pwm_ratio[1]),
	.sd2_pwm_target         (sd_pwm_ratio[2]),
	.sd3_pwm_target         (sd_pwm_ratio[3]),
	.servo0_pwm_target      (servo_pwm_ratio[0]),
	.servo1_pwm_target      (servo_pwm_ratio[1]),
	.servo2_pwm_target      (servo_pwm_ratio[2]),
	.servo3_pwm_target      (servo_pwm_ratio[3]),
	.pwm_update             (pwm_update[11:0]),
	.crc_error              () 
); 

////////////////////////////////////////////////////////////////
// Swerve Rotation Motor
////////////////////////////////////////////////////////////////
pwm_ctrl pwm_ctrl0(
    .reset_n                (reset_n),          // Active low reset
    .clock                  (clock),            // The main clock

    //FPGA Subsystem Interface
    .target_angle           (sr_pwm_target[0]), // The angle the wheel needs to move to in degrees. This number is multiplied by 2 internally
    .angle_update           (pwm_update[0]),    // Signals when an angle update is available
    .angle_done             (pwm_done[0]),      // Output sent when angle has been adjusted to target_angle

    //PWM Interface
    .pwm_done               (),                 // Updated PWM ratio has been applied (1 cycle long pulse)
    .pwm_enable             (sr_pwm_enable[0]), // Enables the PWM output
    .pwm_ratio              (sr_pwm_ratio[0]),  // The high-time of the PWM signal out of 255.
    .pwm_update             (sr_pwm_update[0]), // Request an update to the PWM ratio

    //I2C Interface
    .sck                    (sck[0]),           // The I2C clock
    .sda                    (sda[0])            // The I2C bi-directional data
);

pwm sr_pwm0(
    .reset_n                (reset_n),
    .clock                  (clock),
    .pwm_enable             (sr_pwm_enable[0]),
    .pwm_ratio              (sr_pwm_ratio[0]),
    .pwm_update             (sr_pwm_update[0]), 
    .pwm_signal             (sr_pwm[0])
);

pwm_ctrl pwm_ctrl1(
    .reset_n                (reset_n),          // Active low reset
    .clock                  (clock),            // The main clock

    //FPGA Subsystem Interface
    .target_angle           (sr_pwm_target[1]), // The angle the wheel needs to move to in degrees. This number is multiplied by 2 internally
    .angle_update           (pwm_update[1]),    // Signals when an angle update is available
    .angle_done             (pwm_done[1]),      // Output sent when angle has been adjusted to target_angle

    //PWM Interface
    .pwm_done               (),                 // Updated PWM ratio has been applied (1 cycle long pulse)
    .pwm_enable             (sr_pwm_enable[1]), // Enables the PWM output
    .pwm_ratio              (sr_pwm_ratio[1]),  // The high-time of the PWM signal out of 255.
    .pwm_update             (sr_pwm_update[1]), // Request an update to the PWM ratio

    //I2C Interface
    .sck                    (sck[1]),           // The I2C clock
    .sda                    (sda[1])            // The I2C bi-directional data
);

pwm sr_pwm1(
    .reset_n                (reset_n),
    .clock                  (clock),
    .pwm_enable             (sr_pwm_enable[1]),
    .pwm_ratio              (sr_pwm_ratio[1]),
    .pwm_update             (sr_pwm_update[1]), 
    .pwm_signal             (sr_pwm[1])
);

pwm_ctrl pwm_ctrl2(
    .reset_n                (reset_n),          // Active low reset
    .clock                  (clock),            // The main clock

    //FPGA Subsystem Interface
    .target_angle           (sr_pwm_target[2]), // The angle the wheel needs to move to in degrees. This number is multiplied by 2 internally
    .angle_update           (pwm_update[2]),    // Signals when an angle update is available
    .angle_done             (pwm_done[2]),      // Output sent when angle has been adjusted to target_angle

    //PWM Interface
    .pwm_done               (),                 // Updated PWM ratio has been applied (1 cycle long pulse)
    .pwm_enable             (sr_pwm_enable[2]), // Enables the PWM output
    .pwm_ratio              (sr_pwm_ratio[2]),  // The high-time of the PWM signal out of 255.
    .pwm_update             (sr_pwm_update[2]), // Request an update to the PWM ratio

    //I2C Interface
    .sck                    (sck[2]),           // The I2C clock
    .sda                    (sda[2])            // The I2C bi-directional data
);

pwm sr_pwm2(
    .reset_n                (reset_n),
    .clock                  (clock),
    .pwm_enable             (sr_pwm_enable[2]),
    .pwm_ratio              (sr_pwm_ratio[2]),
    .pwm_update             (sr_pwm_update[2]), 
    .pwm_signal             (sr_pwm[2])
);

pwm_ctrl pwm_ctrl3(
    .reset_n                (reset_n),          // Active low reset
    .clock                  (clock),            // The main clock

    //FPGA Subsystem Interface
    .target_angle           (sr_pwm_target[3]), // The angle the wheel needs to move to in degrees. This number is multiplied by 2 internally
    .angle_update           (pwm_update[3]),    // Signals when an angle update is available
    .angle_done             (pwm_done[3]),      // Output sent when angle has been adjusted to target_angle

    //PWM Interface
    .pwm_done               (),                 // Updated PWM ratio has been applied (1 cycle long pulse)
    .pwm_enable             (sr_pwm_enable[3]), // Enables the PWM output
    .pwm_ratio              (sr_pwm_ratio[3]),  // The high-time of the PWM signal out of 255.
    .pwm_update             (sr_pwm_update[3]), // Request an update to the PWM ratio

    //I2C Interface
    .sck                    (sck[3]),           // The I2C clock
    .sda                    (sda[3])            // The I2C bi-directional data
);

pwm sr_pwm3(
    .reset_n                (reset_n),
    .clock                  (clock),
    .pwm_enable             (sr_pwm_enable[3]),
    .pwm_ratio              (sr_pwm_ratio[3]),
    .pwm_update             (sr_pwm_update[3]), 
    .pwm_signal             (sr_pwm[3])
);


////////////////////////////////////////////////////////////////
// Swerve Drive Motor
////////////////////////////////////////////////////////////////
pwm sd_pwm0(
    .reset_n                (reset_n),
    .clock                  (clock),
    .pwm_enable             (1'b1),
    .pwm_ratio              (sd_pwm_ratio[0]),
    .pwm_update             (1'b1), 
    .pwm_signal             (sd_pwm[0])
);

pwm sd_pwm1(
    .reset_n                (reset_n),
    .clock                  (clock),
    .pwm_enable             (1'b1),
    .pwm_ratio              (sd_pwm_ratio[1]),
    .pwm_update             (1'b1), 
    .pwm_signal             (sd_pwm[1])
);

pwm sd_pwm2(
    .reset_n                (reset_n),
    .clock                  (clock),
    .pwm_enable             (1'b1),
    .pwm_ratio              (sd_pwm_ratio[2]),
    .pwm_update             (1'b1), 
    .pwm_signal             (sd_pwm[2])
);

pwm sd_pwm3(
    .reset_n                (reset_n),
    .clock                  (clock),
    .pwm_enable             (1'b1),
    .pwm_ratio              (sd_pwm_ratio[3]),
    .pwm_update             (1'b1), 
    .pwm_signal             (sd_pwm[3])
);


////////////////////////////////////////////////////////////////
// Servos
////////////////////////////////////////////////////////////////
pwm servo_pwm0(
    .reset_n                (reset_n),
    .clock                  (clock),
    .pwm_enable             (1'b1),
    .pwm_ratio              (servo_pwm_ratio[0]),
    .pwm_update             (1'b1), 
    .pwm_signal             (servo_pwm[0])
);

pwm servo_pwm1(
    .reset_n                (reset_n),
    .clock                  (clock),
    .pwm_enable             (1'b1),
    .pwm_ratio              (servo_pwm_ratio[1]),
    .pwm_update             (1'b1), 
    .pwm_signal             (servo_pwm[1])
);

pwm servo_pwm2(
    .reset_n                (reset_n),
    .clock                  (clock),
    .pwm_enable             (1'b1),
    .pwm_ratio              (servo_pwm_ratio[2]),
    .pwm_update             (1'b1), 
    .pwm_signal             (servo_pwm[2])
);

pwm servo_pwm3(
    .reset_n                (reset_n),
    .clock                  (clock),
    .pwm_enable             (1'b1),
    .pwm_ratio              (servo_pwm_ratio[3]),
    .pwm_update             (1'b1), 
    .pwm_signal             (servo_pwm[3])
);
*/
endmodule