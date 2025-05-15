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

    wire  [4:0]     pwm0;      	            // PWM control  
    wire  [4:0]     pwm1;      	            // PWM control  
    wire  [4:0]     pwm2;      	            // PWM control  
    wire  [4:0]     pwm3;      	            // PWM control
    wire  [3:0]     sr_pwm_done, sr_pwm_enable, sr_pwm_update, sr_pwm_direction;
    wire  [7:0]     sr_pwm_ratio    [3:0];

    wire  [11:0]    target_angle0;          // Rotation target angle
    wire  [11:0]    current_angle0;         // The current angle
    wire  [11:0]    target_angle1;          // Rotation target angle
    wire  [11:0]    current_angle1;         // The current angle
    wire  [11:0]    target_angle2;          // Rotation target angle
    wire  [11:0]    current_angle2;         // The current angle
    wire  [11:0]    target_angle3;          // Rotation target angle
    wire  [11:0]    current_angle3;         // The current angle
    wire            abort_angle0;           // Aborts rotating to angle
    wire            abort_angle1;           // Aborts rotating to angle
    wire            abort_angle2;           // Aborts rotating to angle
    wire            abort_angle3;           // Aborts rotating to angle
    wire            update_angle0;          // Start rotation to angle
    wire            update_angle1;          // Start rotation to angle
    wire            update_angle2;          // Start rotation to angle
    wire            update_angle3;          // Start rotation to angle
    wire            angle_done0;            // Arrived at target angle
    wire            angle_done1;            // Arrived at target angle
    wire            angle_done2;            // Arrived at target angle
    wire            angle_done3;            // Arrived at target angle
    wire  [7:0]     pwm_value0;             // Overridable PWM value
    wire  [7:0]     pwm_value1;             // Overridable PWM value
    wire  [7:0]     pwm_value2;             // Overridable PWM value
    wire  [7:0]     pwm_value3;             // Overridable PWM value
 
    wire  [7:0]     servo_control;          // Servo control   
    wire  [7:0]     servo_position0;        // Servo 0 target position
    wire  [7:0]     servo_position1;        // Servo 1 target position
    wire  [7:0]     servo_position2;        // Servo 2 target position
    wire  [7:0]     servo_position3;        // Servo 3 target position

    wire  [31:0]    debug_signals;          // Debug signals
    wire  [7:0]     startup_fail;
    wire  [15:0]    pwm_ctrl0_debug;        
    wire            led_test_enable;        // Enable the led testing
    wire            pi_connected_led, ps4_connected_led, motor_hot_led, fault_led;
    wire  [3:0]     led_pwm;
    wire  [3:0]     sd_pwm_enable, sd_pwm_direction, sd_i2c_caliben;
    wire            enable_stall_chk0, enable_stall_chk1, enable_stall_chk2, enable_stall_chk3;
    wire  [7:0]     kp0, kp1, kp2, kp3;
    wire  [3:0]     ki0, ki1, ki2, ki3;
    wire  [3:0]     kd0, kd1, kd2, kd3;
    wire            rot_pwm_ovrd0, rot_pwm_ovrd1, rot_pwm_ovrd2, rot_pwm_ovrd3;
    wire            pwm_dir_ovrd0, pwm_dir_ovrd1, pwm_dir_ovrd2, pwm_dir_ovrd3;
    wire  [5:0]     pwm_ratio_ovrd0, pwm_ratio_ovrd1, pwm_ratio_ovrd2, pwm_ratio_ovrd3;

assign startup_fail[3:0] = 4'b0;
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
    .pwm0               (pwm0),      	            // PWM control  
    .pwm1               (pwm1),      	            // PWM control  
    .pwm2               (pwm2),      	            // PWM control  
    .pwm3               (pwm3),      	            // PWM control

    .sr_enable0         (sr_pwm_enable[0]),         // Motor enable
    .sr_enable1         (sr_pwm_enable[1]),         // Motor enable
    .sr_enable2         (sr_pwm_enable[2]),         // Motor enable
    .sr_enable3         (sr_pwm_enable[3]),         // Motor enable
    .sd_enable0         (sd_pwm_enable[0]),         // Motor enable
    .sd_enable1         (sd_pwm_enable[1]),         // Motor enable
    .sd_enable2         (sd_pwm_enable[2]),         // Motor enable
    .sd_enable3         (sd_pwm_enable[3]),         // Motor enable

//    .direction0         (sr_pwm_direction[0]),    // Motor direction
//    .direction1         (sr_pwm_direction[1]),    // Motor direction
//    .direction2         (sr_pwm_direction[2]),    // Motor direction
//    .direction3         (sr_pwm_direction[3]),    // Motor direction
    .i2c_calib_en0      (sd_i2c_caliben[0]),   // I2C enable override for calibration 
    .i2c_calib_en1      (sd_i2c_caliben[1]),   // I2C enable override for calibration 
    .i2c_calib_en2      (sd_i2c_caliben[2]),   // I2C enable override for calibration 
    .i2c_calib_en3      (sd_i2c_caliben[3]),   // I2C enable override for calibration 
    .direction4         (sd_pwm_direction[0]),      // Motor direction
    .direction5         (sd_pwm_direction[1]),      // Motor direction
    .direction6         (sd_pwm_direction[2]),      // Motor direction
    .direction7         (sd_pwm_direction[3]),      // Motor direction

    .startup_fail       (startup_fail[7:0]),
    // Rotation Motors outputs
    .enable_stall_chk0  (enable_stall_chk0),        // Enable the stall check
    .kp0                (kp0[7:0]),                 // Proportional Constant: fixed point 4.4
    .ki0                (ki0[3:0]),                 // Integral Constant: fixed point 0.4
    .kd0                (kd0[3:0]),                 // Derivative Constant: fixed point 0.4
    .rot_pwm_ovrd0      (rot_pwm_ovrd0),            // Rotation motor override enable
    .pwm_dir_ovrd0      (pwm_dir_ovrd0),            // Rotation motor override direction
    .pwm_ratio_ovrd0    (pwm_ratio_ovrd0[5:0]),     // Rotation motor override value
    .enable_stall_chk1  (enable_stall_chk1),        // Enable the stall check
    .kp1                (kp1[7:0]),                 // Proportional Constant: fixed point 4.4
    .ki1                (ki1[3:0]),                 // Integral Constant: fixed point 0.4
    .kd1                (kd1[3:0]),                 // Derivative Constant: fixed point 0.4
    .rot_pwm_ovrd1      (rot_pwm_ovrd1),            // Rotation motor override enable
    .pwm_dir_ovrd1      (pwm_dir_ovrd1),            // Rotation motor override direction
    .pwm_ratio_ovrd1    (pwm_ratio_ovrd1[5:0]),     // Rotation motor override value
    .enable_stall_chk2  (enable_stall_chk2),        // Enable the stall check
    .kp2                (kp2[7:0]),                 // Proportional Constant: fixed point 4.4
    .ki2                (ki2[3:0]),                 // Integral Constant: fixed point 0.4
    .kd2                (kd2[3:0]),                 // Derivative Constant: fixed point 0.4
    .rot_pwm_ovrd2      (rot_pwm_ovrd2),            // Rotation motor override enable
    .pwm_dir_ovrd2      (pwm_dir_ovrd2),            // Rotation motor override direction
    .pwm_ratio_ovrd2    (pwm_ratio_ovrd2[5:0]),     // Rotation motor override value
    .enable_stall_chk3  (enable_stall_chk3),        // Enable the stall check
    .kp3                (kp3[7:0]),                 // Proportional Constant: fixed point 4.4
    .ki3                (ki3[3:0]),                 // Integral Constant: fixed point 0.4
    .kd3                (kd3[3:0]),                 // Derivative Constant: fixed point 0.4
    .rot_pwm_ovrd3      (rot_pwm_ovrd3),            // Rotation motor override enable
    .pwm_dir_ovrd3      (pwm_dir_ovrd3),            // Rotation motor override direction
    .pwm_ratio_ovrd3    (pwm_ratio_ovrd3[5:0]),     // Rotation motor override value

    .target_angle0      (target_angle0),            // Rotation target angle
    .current_angle0     (current_angle0),           // The current angle
    .target_angle1      (target_angle1),            // Rotation target angle
    .current_angle1     (current_angle1),           // The current angle
    .target_angle2      (target_angle2),            // Rotation target angle
    .current_angle2     (current_angle2),           // The current angle
    .target_angle3      (target_angle3),            // Rotation target angle
    .current_angle3     (current_angle3),           // The current angle

    .abort_angle0       (abort_angle0),             // Aborts rotating to angle
    .abort_angle1       (abort_angle1),             // Aborts rotating to angle
    .abort_angle2       (abort_angle2),             // Aborts rotating to angle
    .abort_angle3       (abort_angle3),             // Aborts rotating to angle
    .update_angle0      (update_angle0),            // Start rotation to angle
    .update_angle1      (update_angle1),            // Start rotation to angle
    .update_angle2      (update_angle2),            // Start rotation to angle
    .update_angle3      (update_angle3),            // Start rotation to angle
    .angle_done0        (angle_done0),              // Start rotation to angle
    .angle_done1        (angle_done1),              // Start rotation to angle
    .angle_done2        (angle_done2),              // Start rotation to angle
    .angle_done3        (angle_done3),              // Start rotation to angle

    .servo_position0    (servo_position0),          // Servo 0 target position
    .servo_position1    (servo_position1),          // Servo 1 target position
    .servo_position2    (servo_position2),          // Servo 2 target position
    .servo_position3    (servo_position3),          // Servo 3 target position

    .debug_signals      (debug_signals[31:0]),      // Debug signals
    .led_test_enable    (led_test_enable),          // Enable the led testing
    .pi_connected_led   (pi_connected_led),         // Orange Pi connected
    .ps4_connected_led  (ps4_connected_led),        // PS4 connected
    .fault_led          (fault_led),                // Fault led
    .motor_hot_led      (motor_hot_led)             // Hot motor led
);  

/*
////////////////////////////////////////////////////////////////
// Swerve Rotation Motor0
////////////////////////////////////////////////////////////////
pwm_ctrl pwm_ctrl0(
    .reset_n                (reset_n),              // Active low reset
    .clock                  (clock),                // The main clock

    // FPGA Subsystem Interface
    .target_angle           (target_angle0[11:0]),  // The angle the wheel needs to move to in degrees. This number is multiplied by 2 internally
    .angle_update           (update_angle0),        // Signals when an angle update is available
    .current_angle          (current_angle0[11:0]), // Angle we are currently at from I2C
    .abort_angle            (abort_angle0),         // Aborts the angle adjustment
    .angle_done             (angle_done0),          // Output sent when angle has been adjusted to target_angle

    // Acceleration interface
    .enable_stall_chk       (enable_stall_chk0),    // Enable the stall check
    .stalled                (startup_fail[4]),      // Error: Motor stalled, unable to startup   .debug_signals  (debug_signals[7:0]),

    // PID coefficients
    .kp                     (kp0[7:0]),              // Proportional Constant: fixed point 4.4
    .ki                     (ki0[3:0]),              // Integral Constant: fixed point 0.4
    .kd                     (kd0[3:0]),              // Derivative Constant: fixed point 0.4

    // PWM Interface
    .pwm_done               (sr_pwm_done[0]),       // Updated PWM ratio has been applied (1 cycle long pulse)
    .pwm_enable             (sr_pwm_enable[0]),     // Enables the PWM output
    .calib_en               (sd_i2c_caliben[0]),    // I2C enable override for calibration
    .pwm_ratio              (sr_pwm_ratio[0]),      // The high-time of the PWM signal out of 255.
    .pwm_direction          (sr_pwm_direction[0]),  // The direction of the motor
    .pwm_update             (sr_pwm_update[0]),     // Request an update to the PWM ratio

    .debug_signals          (pwm_ctrl0_debug[15:0]),

    //I2C Interface
    .sck                    (scl[0]),               // The I2C clock
    .sda                    (sda[0])                // The I2C bi-directional data
);

assign debug_signals[31:0] = {  pwm_ctrl0_debug[15:0],  // 31:16
                                sr_pwm_ratio[0][7:0],   // 15:8
                                3'b0, sr_pwm_direction[0], sr_pwm_done[0], sr_pwm_enable[0],  sr_pwm_update[0], angle_done0};

// Register override to control angle motors
assign pwm_value0 = rot_pwm_ovrd0 ? {pwm_ratio_ovrd0[5:0], 2'b0}: sr_pwm_ratio[0];

spark_pwm sr_pwm0(
    .reset_n                (reset_n),                              // Active low reset
    .clock                  (clock_div_cntr[5]),                    // ~422kHz
    .pwm_enable             (sr_pwm_enable[0] | rot_pwm_ovrd0),     // PWM enable
    .pwm_ratio              (pwm_value0),                           // The high-time of the PWM signal out of 255
    .pwm_direction          (sr_pwm_direction[0]),                  // Motor direction
    .pwm_update             (sr_pwm_update[0] | rot_pwm_ovrd0),     // Request an update to the PWM ratio
    .pwm_done               (sr_pwm_done[0]),                       // Updated PWM ratio has been applied (pulse)
    .pwm_signal             (sr_pwm[0])                             // The output PWM wave
);

////////////////////////////////////////////////////////////////
// Swerve Rotation Motor1
////////////////////////////////////////////////////////////////
pwm_ctrl pwm_ctrl1(
    .reset_n                (reset_n),              // Active low reset
    .clock                  (clock),                // The main clock

    // FPGA Subsystem Interface
    .target_angle           (target_angle1[11:0]),  // The angle the wheel needs to move to in degrees. This number is multiplied by 2 internally
    .angle_update           (update_angle1),        // Signals when an angle update is available
    .current_angle          (current_angle1[11:0]), // Angle we are currently at from I2C
    .abort_angle            (abort_angle1),         // Aborts the angle adjustment
    .angle_done             (angle_done1),          // Output sent when angle has been adjusted to target_angle

    // Acceleration interface
    .enable_stall_chk       (enable_stall_chk1),    // Enable the stall check
    .stalled                (startup_fail[5]),      // Error: Motor stalled, unable to startup   .debug_signals  (debug_signals[7:0]),

    // PID coefficients
    .kp                     (kp1[7:0]),              // Proportional Constant: fixed point 4.4
    .ki                     (ki1[3:0]),              // Integral Constant: fixed point 0.4
    .kd                     (kd1[3:0]),              // Derivative Constant: fixed point 0.4

    // PWM Interface
    .pwm_done               (sr_pwm_done[1]),       // Updated PWM ratio has been applied (1 cycle long pulse)
    .pwm_enable             (sr_pwm_enable[1]),     // Enables the PWM output
    .calib_en               (sd_i2c_caliben[1]),    // I2C enable override for calibration
    .pwm_ratio              (sr_pwm_ratio[1]),      // The high-time of the PWM signal out of 255.
    .pwm_direction          (sr_pwm_direction[1]),  // The direction of the motor
    .pwm_update             (sr_pwm_update[1]),     // Request an update to the PWM ratio

    .debug_signals          (),

    //I2C Interface
    .sck                    (scl[1]),               // The I2C clock
    .sda                    (sda[1])                // The I2C bi-directional data
);

// Register override to control angle motors
assign pwm_value1 = rot_pwm_ovrd1 ? {pwm_ratio_ovrd1[5:0], 2'b0}: sr_pwm_ratio[1];

spark_pwm sr_pwm1(
    .reset_n                (reset_n),                              // Active low reset
    .clock                  (clock_div_cntr[5]),                    // ~422kHz
    .pwm_enable             (sr_pwm_enable[1] | rot_pwm_ovrd1),     // PWM enable
    .pwm_ratio              (pwm_value1),                           // The high-time of the PWM signal out of 255
    .pwm_direction          (sr_pwm_direction[1]),                  // Motor direction
    .pwm_update             (sr_pwm_update[1] | rot_pwm_ovrd1),     // Request an update to the PWM ratio
    .pwm_done               (sr_pwm_done[1]),                       // Updated PWM ratio has been applied (pulse)
    .pwm_signal             (sr_pwm[1])                             // The output PWM wave
);

////////////////////////////////////////////////////////////////
// Swerve Rotation Motor2
////////////////////////////////////////////////////////////////
pwm_ctrl pwm_ctrl2(
    .reset_n                (reset_n),              // Active low reset
    .clock                  (clock),                // The main clock

    // FPGA Subsystem Interface
    .target_angle           (target_angle2[11:0]),  // The angle the wheel needs to move to in degrees. This number is multiplied by 2 internally
    .angle_update           (update_angle2),        // Signals when an angle update is available
    .current_angle          (current_angle2[11:0]), // Angle we are currently at from I2C
    .abort_angle            (abort_angle2),         // Aborts the angle adjustment
    .angle_done             (angle_done2),          // Output sent when angle has been adjusted to target_angle

    // Acceleration interface
    .enable_stall_chk       (enable_stall_chk2),    // Enable the stall check
    .stalled                (startup_fail[6]),      // Error: Motor stalled, unable to startup   .debug_signals  (debug_signals[7:0]),

    // PID coefficients
    .kp                     (kp2[7:0]),              // Proportional Constant: fixed point 4.4
    .ki                     (ki2[3:0]),              // Integral Constant: fixed point 0.4
    .kd                     (kd2[3:0]),              // Derivative Constant: fixed point 0.4

    // PWM Interface
    .pwm_done               (sr_pwm_done[2]),       // Updated PWM ratio has been applied (1 cycle long pulse)
    .pwm_enable             (sr_pwm_enable[2]),     // Enables the PWM output
    .calib_en               (sd_i2c_caliben[2]),    // I2C enable override for calibration
    .pwm_ratio              (sr_pwm_ratio[2]),      // The high-time of the PWM signal out of 255.
    .pwm_direction          (sr_pwm_direction[2]),  // The direction of the motor
    .pwm_update             (sr_pwm_update[2]),     // Request an update to the PWM ratio

    .debug_signals          (),

    //I2C Interface
    .sck                    (scl[2]),               // The I2C clock
    .sda                    (sda[2])                // The I2C bi-directional data
);

// Register override to control angle motors
assign pwm_value2 = rot_pwm_ovrd2 ? {pwm_ratio_ovrd2[5:0], 2'b0}: sr_pwm_ratio[2];

spark_pwm sr_pwm2(
    .reset_n                (reset_n),                              // Active low reset
    .clock                  (clock_div_cntr[5]),                    // ~422kHz
    .pwm_enable             (sr_pwm_enable[2] | rot_pwm_ovrd2),     // PWM enable
    .pwm_ratio              (pwm_value2),                           // The high-time of the PWM signal out of 255
    .pwm_direction          (sr_pwm_direction[2]),                  // Motor direction
    .pwm_update             (sr_pwm_update[2] | rot_pwm_ovrd2),     // Request an update to the PWM ratio
    .pwm_done               (sr_pwm_done[2]),                       // Updated PWM ratio has been applied (pulse)
    .pwm_signal             (sr_pwm[2])                             // The output PWM wave
);
*/
////////////////////////////////////////////////////////////////
// Swerve Rotation Motor3
////////////////////////////////////////////////////////////////
pwm_ctrl pwm_ctrl3(
    .reset_n                (reset_n),              // Active low reset
    .clock                  (clock),                // The main clock

    // FPGA Subsystem Interface
    .target_angle           (target_angle3[11:0]),  // The angle the wheel needs to move to in degrees. This number is multiplied by 2 internally
    .angle_update           (update_angle3),        // Signals when an angle update is available
    .current_angle          (current_angle3[11:0]), // Angle we are currently at from I2C
    .abort_angle            (abort_angle3),         // Aborts the angle adjustment
    .angle_done             (angle_done3),          // Output sent when angle has been adjusted to target_angle

    // Acceleration interface
    .enable_stall_chk       (enable_stall_chk3),    // Enable the stall check
    .stalled                (startup_fail[7]),      // Error: Motor stalled, unable to startup   .debug_signals  (debug_signals[7:0]),

    // PID coefficients
    .kp                     (kp3[7:0]),              // Proportional Constant: fixed point 4.4
    .ki                     (ki3[3:0]),              // Integral Constant: fixed point 0.4
    .kd                     (kd3[3:0]),              // Derivative Constant: fixed point 0.4

    // PWM Interface
    .pwm_done               (sr_pwm_done[3]),       // Updated PWM ratio has been applied (1 cycle long pulse)
    .pwm_enable             (sr_pwm_enable[3]),     // Enables the PWM output
    .calib_en               (sd_i2c_caliben[3]),    // I2C enable override for calibration
    .pwm_ratio              (sr_pwm_ratio[3]),      // The high-time of the PWM signal out of 255.
    .pwm_direction          (sr_pwm_direction[3]),  // The direction of the motor
    .pwm_update             (sr_pwm_update[3]),     // Request an update to the PWM ratio

    .debug_signals          (),

    //I2C Interface
    .sck                    (scl[3]),               // The I2C clock
    .sda                    (sda[3])                // The I2C bi-directional data
);

// Register override to control angle motors
assign pwm_value3 = rot_pwm_ovrd3 ? {pwm_ratio_ovrd3[5:0], 2'b0}: sr_pwm_ratio[3];

spark_pwm sr_pwm3(
    .reset_n                (reset_n),                              // Active low reset
    .clock                  (clock_div_cntr[5]),                    // ~422kHz
    .pwm_enable             (sr_pwm_enable[3] | rot_pwm_ovrd3),     // PWM enable
    .pwm_ratio              (pwm_value3),                           // The high-time of the PWM signal out of 255
    .pwm_direction          (sr_pwm_direction[3]),                  // Motor direction
    .pwm_update             (sr_pwm_update[3] | rot_pwm_ovrd3),     // Request an update to the PWM ratio
    .pwm_done               (sr_pwm_done[3]),                       // Updated PWM ratio has been applied (pulse)
    .pwm_signal             (sr_pwm[3])                             // The output PWM wave
);
/*
////////////////////////////////////////////////////////////////
// Swerve Drive Motor0
////////////////////////////////////////////////////////////////
spark_pwm sd_pwm0(
    .reset_n                (reset_n),                  // Active low reset
    .clock                  (clock_div_cntr[5]),        // ~422kHz
    .pwm_enable             (sd_pwm_enable[0]),         // PWM enable
    .pwm_ratio              ({1'b0, pwm0[4:0], 2'b0}),  // The high-time of the PWM signal out of 255
    .pwm_direction          (sd_pwm_direction[0]),      // Motor direction
    .pwm_update             (1'b1),                     // Request an update to the PWM ratio
    .pwm_done               (),                         // Updated PWM ratio has been applied (pulse)
    .pwm_signal             (sd_pwm[0])                 // The output PWM wave
);

////////////////////////////////////////////////////////////////
// Swerve Drive Motor1
////////////////////////////////////////////////////////////////
spark_pwm sd_pwm1(
    .reset_n                (reset_n),                  // Active low reset
    .clock                  (clock_div_cntr[5]),        // ~422kHz
    .pwm_enable             (sd_pwm_enable[1]),         // PWM enable
    .pwm_ratio              ({1'b0, pwm1[4:0], 2'b0}),  // The high-time of the PWM signal out of 255
    .pwm_direction          (sd_pwm_direction[1]),      // Motor direction
    .pwm_update             (1'b1),                     // Request an update to the PWM ratio
    .pwm_done               (),                         // Updated PWM ratio has been applied (pulse)
    .pwm_signal             (sd_pwm[1])                 // The output PWM wave
);

////////////////////////////////////////////////////////////////
// Swerve Drive Motor2
////////////////////////////////////////////////////////////////
spark_pwm sd_pwm2(
    .reset_n                (reset_n),                  // Active low reset
    .clock                  (clock_div_cntr[5]),        // ~422kHz
    .pwm_enable             (sd_pwm_enable[2]),         // PWM enable
    .pwm_ratio              ({1'b0, pwm2[4:0], 2'b0}),  // The high-time of the PWM signal out of 255
    .pwm_direction          (sd_pwm_direction[2]),      // Motor direction
    .pwm_update             (1'b1),                     // Request an update to the PWM ratio
    .pwm_done               (),                         // Updated PWM ratio has been applied (pulse)
    .pwm_signal             (sd_pwm[2])                 // The output PWM wave
);

////////////////////////////////////////////////////////////////
// Swerve Drive Motor3
////////////////////////////////////////////////////////////////
spark_pwm sd_pwm3(
    .reset_n                (reset_n),                  // Active low reset
    .clock                  (clock_div_cntr[5]),        // ~422kHz
    .pwm_enable             (sd_pwm_enable[3]),         // PWM enable
    .pwm_ratio              ({1'b0, pwm3[4:0], 2'b0}),  // The high-time of the PWM signal out of 255
    .pwm_direction          (sd_pwm_direction[3]),      // Motor direction
    .pwm_update             (1'b1),                     // Request an update to the PWM ratio
    .pwm_done               (),                         // Updated PWM ratio has been applied (pulse)
    .pwm_signal             (sd_pwm[3])                 // The output PWM wave
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
*/
endmodule
