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
    output  [3:0]   scl,            // The I2C clock to encoders
    inout   [3:0]   sda,            // The I2C bi-directional data to/from encoders
    output  [7:0]   gnd,
    output  [3:0]   pwr,

    // Swerve Drive Motors
    output  [3:0]   sd_pwm,         // The swerve drive PWM signal

    // Arm Servos
    output  [3:0]   servo_pwm,      // The arm servo PWM wave
    
    // Status and Config
    input           tang_config,    // A 1-bit pull high or low for general configuration
    output          status_fault,   // Control for LED for when a fault has occurred
    output          status_pi,      // Control for LED for when the Orange Pi is connected
    output          status_ps4,     // Control for LED for when the PS4 controller is connected
    output          status_debug    // Control for LED for general debug
);

    reg             reset_n;                // reset
    reg   [2:0]     reset_cntr;             // Reset counter
    reg   [19:0]    clock_div_cntr;         // Clock division counter

    wire  [5:0]     address;   	            // Read / write address
    wire            write_en;  	            // Write enable
    wire  [7:0]     wr_data;   	            // Write data
    wire            read_en;  	            // Read enable
    wire  [7:0]     rd_data;   	            // Read data

    wire            drv_pwm_enable0, drv_pwm_enable1, drv_pwm_enable2, drv_pwm_enable3;
    wire            drv_pwm_dir0, drv_pwm_dir1, drv_pwm_dir2, drv_pwm_dir3;
    wire  [5:0]     drv_pwm_ratio0, drv_pwm_ratio1, drv_pwm_ratio2, drv_pwm_ratio3;

    wire            rot_pwm_enable0, rot_pwm_enable1, rot_pwm_enable2, rot_pwm_enable3;
    wire            rot_pwm_dir0, rot_pwm_dir1, rot_pwm_dir2, rot_pwm_dir3;
    wire  [5:0]     rot_pwm_ratio0, rot_pwm_ratio1, rot_pwm_ratio2, rot_pwm_ratio3;
    wire  [11:0]    current_angle0, current_angle1, current_angle2, current_angle3;

    wire  [7:0]     servo_control;          // Servo control   
    wire  [7:0]     servo_position0;        // Servo 0 target position
    wire  [7:0]     servo_position1;        // Servo 1 target position
    wire  [7:0]     servo_position2;        // Servo 2 target position
    wire  [7:0]     servo_position3;        // Servo 3 target position

    wire  [15:0]    pwm_ctrl0_debug;        
    wire            led_test_enable;        // Enable the led testing
    wire            pi_connected_led, ps4_connected_led, motor_hot_led, fault_led;
    wire  [3:0]     led_pwm;

assign gnd[7:0] = 8'b0;
assign pwr[3:0] = 4'hF;

////////////////////////////////////////////////////////////////
// Reset Controller
////////////////////////////////////////////////////////////////
always @(posedge clock) begin
    if(reset_cntr[2:0] < 3'h7) begin
        reset_cntr[2:0] <= reset_cntr[2:0] + 3'b1;
        reset_n         <= 1'b0;
    end
    else // if reset_cntr == 3'h7
        reset_n <= 1'b1;
end

////////////////////////////////////////////////////////////////
// Clock Division
////////////////////////////////////////////////////////////////
always @(posedge clock or negedge reset_n) begin
    if(~reset_n)
        clock_div_cntr[19:0]     <= 20'b0;
    else
        clock_div_cntr[19:0]     <= clock_div_cntr[19:0] + 20'b1;
end

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
    .reset_n            (reset_n),   	            // Active low reset
    .clock              (clock),     	            // The main clock
    .address            (address[5:0]),	            // Read / write address
    .write_en           (write_en),  	            // Write enable
    .wr_data            (wr_data[7:0]),	            // Write data
    .read_en            (read_en),   	            // Read enable
    .rd_data            (rd_data[7:0]),	            // Read data

    // Drive Motors	outputs	            
    .drv_pwm_enable0    (drv_pwm_enable0),          // Drive motor enable
    .drv_pwm_dir0       (drv_pwm_dir0),             // Drive motor direction
    .drv_pwm_ratio0     (drv_pwm_ratio0),           // Drive motor value

    .drv_pwm_enable1    (drv_pwm_enable1),          // Drive motor enable
    .drv_pwm_dir1       (drv_pwm_dir1),             // Drive motor direction
    .drv_pwm_ratio1     (drv_pwm_ratio1),           // Drive motor value

    .drv_pwm_enable2    (drv_pwm_enable2),          // Drive motor enable
    .drv_pwm_dir2       (drv_pwm_dir2),             // Drive motor direction
    .drv_pwm_ratio2     (drv_pwm_ratio2),           // Drive motor value

    .drv_pwm_enable3    (drv_pwm_enable3),          // Drive motor enable
    .drv_pwm_dir3       (drv_pwm_dir3),             // Drive motor direction
    .drv_pwm_ratio3     (drv_pwm_ratio3),           // Drive motor value
    
    // Rotation Motors outputs
    .rot_pwm_enable0    (rot_pwm_enable0),          // Rotation motor enable
    .rot_pwm_dir0       (rot_pwm_dir0),             // Rotation motor direction
    .rot_pwm_ratio0     (rot_pwm_ratio0),           // Rotation motor value

    .rot_pwm_enable1    (rot_pwm_enable1),          // Rotation motor enable
    .rot_pwm_dir1       (rot_pwm_dir1),             // Rotation motor direction
    .rot_pwm_ratio1     (rot_pwm_ratio1),           // Rotation motor value

    .rot_pwm_enable2    (rot_pwm_enable2),          // Rotation motor enable
    .rot_pwm_dir2       (rot_pwm_dir2),             // Rotation motor direction
    .rot_pwm_ratio2     (rot_pwm_ratio2),           // Rotation motor value

    .rot_pwm_enable3    (rot_pwm_enable3),          // Rotation motor enable
    .rot_pwm_dir3       (rot_pwm_dir3),             // Rotation motor direction
    .rot_pwm_ratio3     (rot_pwm_ratio3),           // Rotation motor value

    // Encoder Values
    .current_angle0     (current_angle0[11:0]),     // AS5600 Raw Angle
    .current_angle1     (current_angle1[11:0]),     // AS5600 Raw Angle
    .current_angle2     (current_angle2[11:0]),     // AS5600 Raw Angle
    .current_angle3     (current_angle3[11:0]),     // AS5600 Raw Angle

    // Servos
    .servo_position0    (servo_position0),          // Servo 0 target position
    .servo_position1    (servo_position1),          // Servo 1 target position
    .servo_position2    (servo_position2),          // Servo 2 target position
    .servo_position3    (servo_position3),          // Servo 3 target position

    // Debug
    .debug_signals      (debug_signals[31:0]),      // Debug signals
    .led_test_enable    (led_test_enable),          // Enable the led testing
    .pi_connected_led   (pi_connected_led),         // Orange Pi connected
    .ps4_connected_led  (ps4_connected_led),        // PS4 connected
    .fault_led          (fault_led),                // Fault led
    .motor_hot_led      (motor_hot_led)             // Hot motor led
);  


////////////////////////////////////////////////////////////////
// Swerve Rotation Motor0
////////////////////////////////////////////////////////////////

i2c i2c_swerve0(    
    .reset_n            (reset_n),              // Active low reset
    .clock              (clk_counter[6]),       // The main clock
    .enable             (rot_pwm_enable0),      // I2C enable override for calibration
    .raw_angle          (current_angle0[11:0]), // The raw angle from the AS5600 
    .rd_done            (),                     // I2C read done pulse           
    .scl                (scl[0]),               // The I2C clock
    .sda                (sda[0])                // The I2C bi-directional data
);

spark_pwm sr_pwm0(
    .reset_n                (reset_n),                          // Active low reset
    .clock                  (clock_div_cntr[5]),                // ~422kHz
    .pwm_enable             (rot_pwm_enable0),                  // PWM enable
    .pwm_ratio              ({rot_pwm_ratio0[5:0], 2'b0}),      // The high-time of the PWM signal out of 255
    .pwm_direction          (rot_pwm_dir0),                     // Motor direction
    .pwm_update             (rot_pwm_enable0),                  // Request an update to the PWM ratio
    .pwm_done               (),                                 // Updated PWM ratio has been applied (pulse)
    .pwm_signal             (sr_pwm[0])                         // The output PWM wave
);


////////////////////////////////////////////////////////////////
// Swerve Rotation Motor1
////////////////////////////////////////////////////////////////

i2c i2c_swerve1(    
    .reset_n            (reset_n),              // Active low reset
    .clock              (clk_counter[6]),       // The main clock
    .enable             (rot_pwm_enable1),      // I2C enable override for calibration
    .raw_angle          (current_angle1[11:0]), // The raw angle from the AS5600 
    .rd_done            (),                     // I2C read done pulse           
    .scl                (scl[1]),               // The I2C clock
    .sda                (sda[1])                // The I2C bi-directional data
);

spark_pwm sr_pwm1(
    .reset_n                (reset_n),                          // Active low reset
    .clock                  (clock_div_cntr[5]),                // ~422kHz
    .pwm_enable             (rot_pwm_enable1),                  // PWM enable
    .pwm_ratio              ({rot_pwm_ratio1[5:0], 2'b0}),      // The high-time of the PWM signal out of 255
    .pwm_direction          (rot_pwm_dir1),                     // Motor direction
    .pwm_update             (rot_pwm_enable1),                  // Request an update to the PWM ratio
    .pwm_done               (),                                 // Updated PWM ratio has been applied (pulse)
    .pwm_signal             (sr_pwm[1])                         // The output PWM wave
);


////////////////////////////////////////////////////////////////
// Swerve Rotation Motor2
////////////////////////////////////////////////////////////////

i2c i2c_swerve2(    
    .reset_n            (reset_n),              // Active low reset
    .clock              (clk_counter[6]),       // The main clock
    .enable             (rot_pwm_enable2),      // I2C enable override for calibration
    .raw_angle          (current_angle2[11:0]), // The raw angle from the AS5600 
    .rd_done            (),                     // I2C read done pulse           
    .scl                (scl[2]),               // The I2C clock
    .sda                (sda[2])                // The I2C bi-directional data
);

spark_pwm sr_pwm2(
    .reset_n                (reset_n),                          // Active low reset
    .clock                  (clock_div_cntr[5]),                // ~422kHz
    .pwm_enable             (rot_pwm_enable2),                  // PWM enable
    .pwm_ratio              ({rot_pwm_ratio2[5:0], 2'b0}),      // The high-time of the PWM signal out of 255
    .pwm_direction          (rot_pwm_dir2),                     // Motor direction
    .pwm_update             (rot_pwm_enable2),                  // Request an update to the PWM ratio
    .pwm_done               (),                                 // Updated PWM ratio has been applied (pulse)
    .pwm_signal             (sr_pwm[2])                         // The output PWM wave
);


////////////////////////////////////////////////////////////////
// Swerve Rotation Motor3
////////////////////////////////////////////////////////////////

i2c i2c_swerve3(    
    .reset_n            (reset_n),              // Active low reset
    .clock              (clk_counter[6]),       // The main clock
    .enable             (rot_pwm_enable3),      // I2C enable override for calibration
    .raw_angle          (current_angle3[11:0]), // The raw angle from the AS5600 
    .rd_done            (),                     // I2C read done pulse           
    .scl                (scl[3]),               // The I2C clock
    .sda                (sda[3])                // The I2C bi-directional data
);

spark_pwm sr_pwm3(
    .reset_n                (reset_n),                          // Active low reset
    .clock                  (clock_div_cntr[5]),                // ~422kHz
    .pwm_enable             (rot_pwm_enable3),                  // PWM enable
    .pwm_ratio              ({rot_pwm_ratio3[5:0], 2'b0}),      // The high-time of the PWM signal out of 255
    .pwm_direction          (rot_pwm_dir3),                     // Motor direction
    .pwm_update             (rot_pwm_enable3),                  // Request an update to the PWM ratio
    .pwm_done               (),                                 // Updated PWM ratio has been applied (pulse)
    .pwm_signal             (sr_pwm[3])                         // The output PWM wave
);


////////////////////////////////////////////////////////////////
// Swerve Drive Motor0
////////////////////////////////////////////////////////////////
spark_pwm sd_pwm0(
    .reset_n                (reset_n),                      // Active low reset
    .clock                  (clock_div_cntr[5]),            // ~422kHz
    .pwm_enable             (drv_pwm_enable0),              // PWM enable
    .pwm_ratio              ({2'b0, drv_pwm_ratio0[5:0]}),  // The high-time of the PWM signal out of 255
    .pwm_direction          (drv_pwm_dir0),                 // Motor direction
    .pwm_update             (1'b1),                         // Request an update to the PWM ratio
    .pwm_done               (),                             // Updated PWM ratio has been applied (pulse)
    .pwm_signal             (sd_pwm[0])                     // The output PWM wave
);

////////////////////////////////////////////////////////////////
// Swerve Drive Motor1
////////////////////////////////////////////////////////////////
spark_pwm sd_pwm1(
    .reset_n                (reset_n),                      // Active low reset
    .clock                  (clock_div_cntr[5]),            // ~422kHz
    .pwm_enable             (drv_pwm_enable1),              // PWM enable
    .pwm_ratio              ({2'b0, drv_pwm_ratio1[5:0]}),  // The high-time of the PWM signal out of 255
    .pwm_direction          (drv_pwm_dir1),                 // Motor direction
    .pwm_update             (1'b1),                         // Request an update to the PWM ratio
    .pwm_done               (),                             // Updated PWM ratio has been applied (pulse)
    .pwm_signal             (sd_pwm[1])                     // The output PWM wave
);

////////////////////////////////////////////////////////////////
// Swerve Drive Motor2
////////////////////////////////////////////////////////////////
spark_pwm sd_pwm2(
    .reset_n                (reset_n),                      // Active low reset
    .clock                  (clock_div_cntr[5]),            // ~422kHz
    .pwm_enable             (drv_pwm_enable2),              // PWM enable
    .pwm_ratio              ({2'b0, drv_pwm_ratio2[5:0]}),  // The high-time of the PWM signal out of 255
    .pwm_direction          (drv_pwm_dir2),                 // Motor direction
    .pwm_update             (1'b1),                         // Request an update to the PWM ratio
    .pwm_done               (),                             // Updated PWM ratio has been applied (pulse)
    .pwm_signal             (sd_pwm[2])                     // The output PWM wave
);

////////////////////////////////////////////////////////////////
// Swerve Drive Motor3
////////////////////////////////////////////////////////////////
spark_pwm sd_pwm3(
    .reset_n                (reset_n),                      // Active low reset
    .clock                  (clock_div_cntr[5]),            // ~422kHz
    .pwm_enable             (drv_pwm_enable3),              // PWM enable
    .pwm_ratio              ({2'b0, drv_pwm_ratio3[5:0]}),  // The high-time of the PWM signal out of 255
    .pwm_direction          (drv_pwm_dir3),                 // Motor direction
    .pwm_update             (1'b1),                         // Request an update to the PWM ratio
    .pwm_done               (),                             // Updated PWM ratio has been applied (pulse)
    .pwm_signal             (sd_pwm[3])                     // The output PWM wave
);


////////////////////////////////////////////////////////////////
// Servos
////////////////////////////////////////////////////////////////
servo_ctrl base_servo(
    .reset_n                (reset_n),
    .clock                  (clock_div_cntr[10]),
    .pwm_enable             (servo_control[0]),
    .start_pwm_ratio        (8'd5),
    .target_pwm_ratio       (servo_position0[7:0]),
    .pwm_signal             (servo_pwm[0])
);

pwm wrist_servo(
    .reset_n                (reset_n),
    .clock                  (clock_div_cntr[10]),
    .pwm_enable             (servo_control[1]),
    .pwm_ratio              (servo_position1[7:0]),
    .pwm_update             (1'b1), 
    .pwm_done               (),
    .pwm_signal             (servo_pwm[1])
);

servo_ctrl center_servo(
    .reset_n                (reset_n),
    .clock                  (clock_div_cntr[10]),
    .pwm_enable             (servo_control[2]),
    .start_pwm_ratio        (8'd5),
    .target_pwm_ratio       (servo_position2[7:0]),
    .pwm_signal             (servo_pwm[2])
);

pwm grabber_servo(
    .reset_n                (reset_n),
    .clock                  (clock_div_cntr[10]),
    .pwm_enable             (servo_control[3]),
    .pwm_ratio              (servo_position3[7:0]),
    .pwm_update             (1'b1), 
    .pwm_done               (),
    .pwm_signal             (servo_pwm[3])
);



////////////////////////////////////////////////////////////////
// LEDs
////////////////////////////////////////////////////////////////
wire pwm_light;
pwm lights(
    .reset_n                (reset_n),              // Active low reset
    .clock                  (clock_div_cntr[10]),   // 
    .pwm_enable             (1'b1),                 // PWM enable
    .pwm_ratio              ({led_pwm[3:0], 4'b0}), // The high-time of the PWM signal out of 255
    .pwm_update             (1'b1),                 // Request an update to the PWM ratio
    .pwm_done               (),                     // Updated PWM ratio has been applied (pulse)
    .pwm_signal             (pwm_light)             // The output PWM wave
);

assign status_fault = pwm_light & fault_led;
assign status_pi    = pwm_light & pi_connected_led;
assign status_ps4   = pwm_light & ps4_connected_led;
assign status_debug = pwm_light;//led_test_enable ? motor_hot_led   : 1'b1;               // Control for LED for general debug

endmodule
