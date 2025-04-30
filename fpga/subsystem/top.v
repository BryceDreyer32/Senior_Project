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
    wire  [7:0]     servo_control;          // Servo control   
    wire  [7:0]     servo_position0;        // Servo 0 target position
    wire  [7:0]     servo_position1;        // Servo 1 target position
    wire  [7:0]     servo_position2;        // Servo 2 target position
    wire  [7:0]     servo_position3;        // Servo 3 target position

    wire  [31:0]    debug_signals;          // Debug signals
    wire  [15:0]    pwm_ctrl0_debug;        
    wire            led_test_enable;        // Enable the led testing
    wire            pi_connected_led, ps4_connected_led, motor_hot_led, fault_led;
    wire  [63:0]    angle_chg;              // Change in angle
    wire  [7:0]     pwm_debug_value;
    wire  [7:0]     kp;
    wire  [3:0]     ki, kd;
    wire            rot_pwm_ovrd;       // Rotation motor override enable
    wire            pwm_dir_ovrd;       // Rotation motor override direction
    wire  [5:0]     pwm_ratio_ovrd;     // Rotation motor override value

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
    .reset_n            (reset_n),   	    // Active low reset
    .clock              (clock),     	    // The main clock
    .address            (address[5:0]),	    // Read / write address
    .write_en           (write_en),  	    // Write enable
    .wr_data            (wr_data[7:0]),	    // Write data
    .read_en            (read_en),   	    // Read enable
    .rd_data            (rd_data[7:0]),	    // Read data
				     
    // Drive Motors	outputs	     
    .pwm0               (pwm0),      	    // PWM control  
    .pwm1               (pwm1),      	    // PWM control  
    .pwm2               (pwm2),      	    // PWM control  
    .pwm3               (pwm3),      	    // PWM control

    // Rotation Motors outputs
    .startup_fail       (startup_fail[7:0]),// Error: Motor stalled, unable to startup
    .enable_stall_chk   (enable_stall_chk), // Enable the stall check

    .target_angle0      (target_angle0),    // Rotation target angle
    .current_angle0     (current_angle0),   // The current angle
    .target_angle1      (target_angle1),    // Rotation target angle
    .current_angle1     (current_angle1),   // The current angle
    .target_angle2      (target_angle2),    // Rotation target angle
    .current_angle2     (current_angle2),   // The current angle
    .target_angle3      (target_angle3),    // Rotation target angle
    .current_angle3     (current_angle3),   // The current angle
    .kp                 (kp),               // Proportional Constant: fixed point 4.4
    .ki                 (ki),               // Integral Constant: fixed point 0.4
    .kd                 (kd),               // Derivative Constant: fixed point 0.4
    .rot_pwm_ovrd       (rot_pwm_ovrd),     // Rotation motor override enable
    .pwm_dir_ovrd       (pwm_dir_ovrd),     // Rotation motor override direction
    .pwm_ratio_ovrd     (pwm_ratio_ovrd),   // Rotation motor override value

    .abort_angle0       (abort_angle0),     // Aborts rotating to angle
    .abort_angle1       (abort_angle1),     // Aborts rotating to angle
    .abort_angle2       (abort_angle2),     // Aborts rotating to angle
    .abort_angle3       (abort_angle3),     // Aborts rotating to angle
    .update_angle0      (update_angle0),    // Start rotation to angle
    .update_angle1      (update_angle1),    // Start rotation to angle
    .update_angle2      (update_angle2),    // Start rotation to angle
    .update_angle3      (update_angle3),    // Start rotation to angle
    .angle_done0        (angle_done0),      // Start rotation to angle
    .angle_done1        (angle_done1),      // Start rotation to angle
    .angle_done2        (angle_done2),      // Start rotation to angle
    .angle_done3        (angle_done3),      // Start rotation to angle

    .servo_control      (servo_control),    // Servo control
    .servo_position0    (servo_position0),  // Servo 0 target position
    .servo_position1    (servo_position1),  // Servo 1 target position
    .servo_position2    (servo_position2),  // Servo 2 target position
    .servo_position3    (servo_position3),  // Servo 3 target position

    .debug_signals      (debug_signals[31:0]),  // Debug signals
    .led_test_enable    (led_test_enable),      // Enable the led testing
    .pi_connected_led   (pi_connected_led),     // Orange Pi connected
    .ps4_connected_led  (ps4_connected_led),    // PS4 connected
    .fault_led          (fault_led),            // Fault led
    .motor_hot_led      (motor_hot_led),        // Hot motor led
    .pwm_debug_value    (pwm_debug_value[7:0])
);

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
    .enable_stall_chk       (enable_stall_chk),     // Enable the stall check
    .stalled                (startup_fail[4]),      // Error: Motor stalled, unable to startup   .debug_signals  (debug_signals[7:0]),

    // PID coefficients
    .kp                     (kp[7:0]),              // Proportional Constant: fixed point 4.4
    .ki                     (ki[3:0]),              // Integral Constant: fixed point 0.4
    .kd                     (kd[3:0]),              // Derivative Constant: fixed point 0.4

    // PWM Interface
    .pwm_done               (sr_pwm_done[0]),       // Updated PWM ratio has been applied (1 cycle long pulse)
    .pwm_enable             (sr_enable[0]),         // Enables the PWM output
    .pwm_ratio              (sr_pwm_ratio[0]),      // The high-time of the PWM signal out of 255.
    .pwm_direction          (sr_direction[0]),      // The direction of the motor
    .pwm_update             (sr_pwm_update[0]),     // Request an update to the PWM ratio

    .debug_signals          (pwm_ctrl0_debug[15:0]),

    //I2C Interface
    .sck                    (scl[0]),               // The I2C clock
    .sda                    (sda[0])                // The I2C bi-directional data
);

wire [7:0] offset;
assign offset = (current_angle0[11:0] < {upperbound1, 4'b0}) & (current_angle0[11:0] > {lowerbound1, 4'b0}) |
                (current_angle0[11:0] < {upperbound2, 4'b0}) & (current_angle0[11:0] > {lowerbound2, 4'b0}) ?
                boost[7:0] : 8'h0;

assign debug_signals[31:0] = {  pwm_ctrl0_debug[15:0],  // 31:16
                                sr_pwm_ratio[0][7:0],   // 15:8
                                3'b0, sr_pwm_direction[0], sr_pwm_done[0], sr_enable[0],  sr_pwm_update[0], angle_done0};

// Register override to control angle motors
wire [7:0]  pwm_value0;
wire        pwm_dir0;
assign pwm_value0 = rot_pwm_ovrd ? {pwm_ratio_ovrd[5:0], 2'b0} - offset: sr_pwm_ratio[0];
assign pwm_dir0   = rot_pwm_ovrd ? pwm_dir_ovrd                 : sr_direction[0];

spark_pwm sr_pwm0(
    .reset_n                (reset_n),              // Active low reset
    .clock                  (clock_div_cntr[5]),    // ~422kHz
    .pwm_enable             (sr_enable[0] | rot_pwm_ovrd),         // PWM enable
    .pwm_ratio              (pwm_value0),           // The high-time of the PWM signal out of 255
    .pwm_direction          (pwm_dir0),             // Motor direction
    .pwm_update             (sr_pwm_update[0] | rot_pwm_ovrd),     // Request an update to the PWM ratio
    .pwm_done               (sr_pwm_done[0]),       // Updated PWM ratio has been applied (pulse)
    .pwm_signal             (sr_pwm[0])             // The output PWM wave
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

//assign status_fault = led_test_enable ? fault_led       : |(sr_fault[3:0]);   // Control for LED for when a fault has occurred
assign status_pi    = led_test_enable ? 1'b1            : pi_connected_led;   // Control for LED for when the Orange Pi is connected
assign status_ps4   = led_test_enable ? 1'b1            : ps4_connected_led;  // Control for LED for when the PS4 controller is connected
assign status_debug = led_test_enable ? motor_hot_led   : 1'b1;               // Control for LED for general debug

endmodule
