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

    wire            reset_n = 1'b1;  // reset

    wire  [5:0]     address;   	     // Read / write address
    wire            write_en;  	     // Write enable
    wire  [7:0]     wr_data;   	     // Write data
    wire            read_en;  	     // Read enable
    wire  [7:0]     rd_data;   	     // Read data

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
    wire  [3:0]     sr_pwm_done, sr_pwm_enable, sr_pwm_update;
    wire  [7:0]     sr_pwm_ratio    [3:0];

    wire  [11:0]    target_angle0;   // Rotation target angle
    wire  [11:0]    current_angle0;  // The current angle
    wire  [11:0]    target_angle1;   // Rotation target angle
    wire  [11:0]    current_angle1;  // The current angle
    wire  [11:0]    target_angle2;   // Rotation target angle
    wire  [11:0]    current_angle2;  // The current angle
    wire  [11:0]    target_angle3;   // Rotation target angle
    wire  [11:0]    current_angle3;  // The current angle

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
				     
    // Drive Motors	inputs: Don't have UART sending back data currently, so strapping for now	     
    .fault0             (1'b0),    	        // Fault signal from motor
    .adc_temp0          (7'b0), 	        // Adc temperature from motor
    .fault1             (1'b0),    	        // Fault signal from motor
    .adc_temp1          (7'b0), 	        // Adc temperature from motor
    .fault2             (1'b0),    	        // Fault signal from motor
    .adc_temp2          (7'b0), 	        // Adc temperature from motor
    .fault3             (1'b0),    	        // Fault signal from motor
    .adc_temp3          (7'b0), 	        // Adc temperature from motor

    // Rotation Motors inputs: Don't have a way to Analog to Digital convert temperature     
    .fault4             (sr_fault[0]),    	// Fault signal from motor
    .adc_temp4          (7'b0), 	        // Adc temperature from motor
    .fault5             (sr_fault[1]), 	    // Fault signal from motor
    .adc_temp5          (7'b0), 	        // Adc temperature from motor
    .fault6             (sr_fault[2]), 	    // Fault signal from motor
    .adc_temp6          (7'b0), 	        // Adc temperature from motor
    .fault7             (sr_fault[3]), 	    // Fault signal from motor
    .adc_temp7          (7'b0), 	        // Adc temperature from motor

    // Drive Motors	outputs	     
	.brake0             (brake0),    	    // Brake control
    .enable0            (enable0),   	    // Motor enable
    .direction0         (direction0),	    // Motor direction
    .pwm0               (pwm0),      	    // PWM control  
    .brake1             (brake1),    	    // Brake control
    .enable1            (enable1),   	    // Motor enable
    .direction1         (direction1),	    // Motor direction
    .pwm1               (pwm1),      	    // PWM control  
    .brake2             (brake2),    	    // Brake control
    .enable2            (enable2),   	    // Motor enable
    .direction2         (direction2),	    // Motor direction
    .pwm2               (pwm2),      	    // PWM control  
    .brake3             (brake3),    	    // Brake control
    .enable3            (enable3),   	    // Motor enable
    .direction3         (direction3),	    // Motor direction
    .pwm3               (pwm3),      	    // PWM control

    // Rotation Motors outputs
    .brake4             (sr_brake[0]),    	// Brake control
    .enable4            (sr_enable[0]),   	// Motor enable
    .direction4         (sr_direction[0]),	// Motor direction
    .brake5             (sr_brake[1]),    	// Brake control
    .enable5            (sr_enable[1]),   	// Motor enable
    .direction5         (sr_direction[1]),	// Motor direction
    .brake6             (sr_brake[2]),    	// Brake control
    .enable6            (sr_enable[2]),   	// Motor enable
    .direction6         (sr_direction[2]),	// Motor direction 
    .brake7             (sr_brake[3]),    	// Brake control
    .enable7            (sr_enable[3]),   	// Motor enable
    .direction7         (sr_direction[3]),	// Motor direction  
					 
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

////////////////////////////////////////////////////////////////
// UART Drive Motor1
////////////////////////////////////////////////////////////////
uart uart_drv_1(
    .reset_n            (reset_n),              // Active low reset
    .clock              (clock),                // The main clock
    .tx_data            ({  brake1, 
                            enable1,
                            direction1,
                            pwm1[4:0]}),        // This the 8-bits of data
    .baud_division      (BAUD_DIVISION[7:0]),   // The division ratio to achieve the desired baud rate
    .tx_start           (1'b1),                 // Signal to indicate that the transmission needs to start
    .uart_tx            (sd_uart[1])            // UART_TX
);

////////////////////////////////////////////////////////////////
// UART Drive Motor2
////////////////////////////////////////////////////////////////
uart uart_drv_2(
    .reset_n            (reset_n),              // Active low reset
    .clock              (clock),                // The main clock
    .tx_data            ({  brake2, 
                            enable2,
                            direction2,
                            pwm2[4:0]}),        // This the 8-bits of data
    .baud_division      (BAUD_DIVISION[7:0]),   // The division ratio to achieve the desired baud rate
    .tx_start           (1'b1),                 // Signal to indicate that the transmission needs to start
    .uart_tx            (sd_uart[2])            // UART_TX
);

////////////////////////////////////////////////////////////////
// UART Drive Motor3
////////////////////////////////////////////////////////////////
uart uart_drv_3(
    .reset_n            (reset_n),              // Active low reset
    .clock              (clock),                // The main clock
    .tx_data            ({  brake3, 
                            enable3,
                            direction3,
                            pwm3[4:0]}),        // This the 8-bits of data
    .baud_division      (BAUD_DIVISION[7:0]),   // The division ratio to achieve the desired baud rate
    .tx_start           (1'b1),                 // Signal to indicate that the transmission needs to start
    .uart_tx            (sd_uart[3])            // UART_TX
);


////////////////////////////////////////////////////////////////
// Swerve Rotation Motor0
////////////////////////////////////////////////////////////////
pwm_ctrl pwm_ctrl0(
    .reset_n                (reset_n),              // Active low reset
    .clock                  (clock),                // The main clock

    //FPGA Subsystem Interface
    .target_angle           (target_angle0[11:0]),  // The angle the wheel needs to move to in degrees. This number is multiplied by 2 internally
    .angle_update           (1'b1),                 // Signals when an angle update is available
    .current_angle          (current_angle0[11:0]), // Angle we are currently at from I2C

    //PWM Interface
    .pwm_done               (sr_pwm_done[0]),       // Updated PWM ratio has been applied (1 cycle long pulse)
    .pwm_enable             (sr_pwm_enable[0]),     // Enables the PWM output
    .pwm_ratio              (sr_pwm_ratio[0]),      // The high-time of the PWM signal out of 255.
    .pwm_update             (sr_pwm_update[0]),     // Request an update to the PWM ratio

    //I2C Interface
    .sck                    (scl[0]),               // The I2C clock
    .sda                    (sda[0])                // The I2C bi-directional data
);

pwm sr_pwm0(
    .reset_n                (reset_n),              // Active low reset
    .clock                  (clock),                // The main clock
    .pwm_enable             (sr_pwm_enable[0]),     // PWM enable
    .pwm_ratio              (sr_pwm_ratio[0]),      // The high-time of the PWM signal out of 255.
    .pwm_update             (sr_pwm_update[0]),     // Request an update to the PWM ratio
    .pwm_done               (sr_pwm_done[0]),       // Updated PWM ratio has been applied (pulse)
    .pwm_signal             (sr_pwm[0])             // The output PWM wave
);

////////////////////////////////////////////////////////////////
// Swerve Rotation Motor1
////////////////////////////////////////////////////////////////
pwm_ctrl pwm_ctrl1(
    .reset_n                (reset_n),              // Active low reset
    .clock                  (clock),                // The main clock

    //FPGA Subsystem Interface
    .target_angle           (target_angle1[11:0]),  // The angle the wheel needs to move to in degrees. This number is multiplied by 2 internally
    .angle_update           (1'b1),                 // Signals when an angle update is available
    .current_angle          (current_angle1[11:0]), // Angle we are currently at from I2C

    //PWM Interface
    .pwm_done               (sr_pwm_done[1]),       // Updated PWM ratio has been applied (1 cycle long pulse)
    .pwm_enable             (sr_pwm_enable[1]),     // Enables the PWM output
    .pwm_ratio              (sr_pwm_ratio[1]),      // The high-time of the PWM signal out of 255.
    .pwm_update             (sr_pwm_update[1]),     // Request an update to the PWM ratio

    //I2C Interface
    .sck                    (scl[1]),               // The I2C clock
    .sda                    (sda[1])                // The I2C bi-directional data
);

pwm sr_pwm1(
    .reset_n                (reset_n),              // Active low reset
    .clock                  (clock),                // The main clock
    .pwm_enable             (sr_pwm_enable[1]),     // PWM enable
    .pwm_ratio              (sr_pwm_ratio[1]),      // The high-time of the PWM signal out of 255.
    .pwm_update             (sr_pwm_update[1]),     // Request an update to the PWM ratio
    .pwm_done               (sr_pwm_done[1]),       // Updated PWM ratio has been applied (pulse)
    .pwm_signal             (sr_pwm[1])             // The output PWM wave
);

////////////////////////////////////////////////////////////////
// Swerve Rotation Motor2
////////////////////////////////////////////////////////////////
pwm_ctrl pwm_ctrl2(
    .reset_n                (reset_n),              // Active low reset
    .clock                  (clock),                // The main clock

    //FPGA Subsystem Interface
    .target_angle           (target_angle2[11:0]),  // The angle the wheel needs to move to in degrees. This number is multiplied by 2 internally
    .angle_update           (1'b1),                 // Signals when an angle update is available
    .current_angle          (current_angle2[11:0]), // Angle we are currently at from I2C

    //PWM Interface
    .pwm_done               (sr_pwm_done[2]),       // Updated PWM ratio has been applied (1 cycle long pulse)
    .pwm_enable             (sr_pwm_enable[2]),     // Enables the PWM output
    .pwm_ratio              (sr_pwm_ratio[2]),      // The high-time of the PWM signal out of 255.
    .pwm_update             (sr_pwm_update[2]),     // Request an update to the PWM ratio

    //I2C Interface
    .sck                    (scl[2]),               // The I2C clock
    .sda                    (sda[2])                // The I2C bi-directional data
);

pwm sr_pwm2(
    .reset_n                (reset_n),              // Active low reset
    .clock                  (clock),                // The main clock
    .pwm_enable             (sr_pwm_enable[2]),     // PWM enable
    .pwm_ratio              (sr_pwm_ratio[2]),      // The high-time of the PWM signal out of 255.
    .pwm_update             (sr_pwm_update[2]),     // Request an update to the PWM ratio
    .pwm_done               (sr_pwm_done[2]),       // Updated PWM ratio has been applied (pulse)
    .pwm_signal             (sr_pwm[2])             // The output PWM wave
);

////////////////////////////////////////////////////////////////
// Swerve Rotation Motor3
////////////////////////////////////////////////////////////////
pwm_ctrl pwm_ctrl3(
    .reset_n                (reset_n),              // Active low reset
    .clock                  (clock),                // The main clock

    //FPGA Subsystem Interface
    .target_angle           (target_angle3[11:0]),  // The angle the wheel needs to move to in degrees. This number is multiplied by 2 internally
    .angle_update           (1'b1),                 // Signals when an angle update is available
    .current_angle          (current_angle3[11:0]), // Angle we are currently at from I2C

    //PWM Interface
    .pwm_done               (sr_pwm_done[3]),       // Updated PWM ratio has been applied (1 cycle long pulse)
    .pwm_enable             (sr_pwm_enable[3]),     // Enables the PWM output
    .pwm_ratio              (sr_pwm_ratio[3]),      // The high-time of the PWM signal out of 255.
    .pwm_update             (sr_pwm_update[3]),     // Request an update to the PWM ratio

    //I2C Interface
    .sck                    (scl[3]),               // The I2C clock
    .sda                    (sda[3])                // The I2C bi-directional data
);

pwm sr_pwm3(
    .reset_n                (reset_n),              // Active low reset
    .clock                  (clock),                // The main clock
    .pwm_enable             (sr_pwm_enable[3]),     // PWM enable
    .pwm_ratio              (sr_pwm_ratio[3]),      // The high-time of the PWM signal out of 255.
    .pwm_update             (sr_pwm_update[3]),     // Request an update to the PWM ratio
    .pwm_done               (sr_pwm_done[3]),       // Updated PWM ratio has been applied (pulse)
    .pwm_signal             (sr_pwm[3])             // The output PWM wave
);

////////////////////////////////////////////////////////////////
// Servos
////////////////////////////////////////////////////////////////
pwm servo_pwm0(
    .reset_n                (reset_n),
    .clock                  (clock),
    .pwm_enable             (1'b1),
    .pwm_ratio              (servo_position0[7:0]),
    .pwm_update             (1'b1),
    .pwm_done               (),
    .pwm_signal             (servo_pwm[0])
);

pwm servo_pwm1(
    .reset_n                (reset_n),
    .clock                  (clock),
    .pwm_enable             (1'b1),
    .pwm_ratio              (servo_position1[7:0]),
    .pwm_update             (1'b1), 
    .pwm_done               (),
    .pwm_signal             (servo_pwm[1])
);

pwm servo_pwm2(
    .reset_n                (reset_n),
    .clock                  (clock),
    .pwm_enable             (1'b1),
    .pwm_ratio              (servo_position2[7:0]),
    .pwm_update             (1'b1), 
    .pwm_done               (),
    .pwm_signal             (servo_pwm[2])
);

pwm servo_pwm3(
    .reset_n                (reset_n),
    .clock                  (clock),
    .pwm_enable             (1'b1),
    .pwm_ratio              (servo_position3[7:0]),
    .pwm_update             (1'b1), 
    .pwm_done               (),
    .pwm_signal             (servo_pwm[3])
);

assign status_fault = |(sr_fault[3:0]);        // Control for LED for when a fault has occurred
assign status_pi = 1'b1;                // Control for LED for when the Orange Pi is connected
assign status_ps4 = 1'b1;               // Control for LED for when the PS4 controller is connected
assign status_debug = 1'b1;             // Control for LED for general debug

endmodule
