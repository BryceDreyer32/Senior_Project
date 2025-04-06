// Copyright 2024
// Bryce's Senior Project
// Description: This is the RTL for PWM Control

module pwm_ctrl(
    input                   reset_n,            // Active low reset
    input                   clock,              // The main clock

    // FPGA Subsystem Interface 
    input           [11:0]  target_angle,       // The angle the wheel needs to move to in degrees. This number is multiplied by 2 internally
    input                   angle_update,       // Signals when an angle update is available
    output                  angle_done,         // Output sent when angle has been adjusted to target_angle
    output  reg     [11:0]  current_angle,      // Angle we are currently at from I2C
    input                   pwm_enable,         // Enables the PWM output
    input                   abort_angle,        // Aborts rotating to angle

    // Acceleration hammer interface    
    input                   enable_stall_chk,   // Enable the stall check
    input           [2:0]   consec_chg,         // Number of consecutive changes we want to see before claiming success
    input           [7:0]   delay_target,       // Number of times to remain on each profile step
    input           [7:0]   profile_offset,     // An offset that is added to each of the profile steps
    input           [7:0]   cruise_power,       // The amount of power to apply during the cruise phase
    output                  startup_fail,       // Error: Motor stalled, unable to startup
    output          [63:0]  angle_chg,          // Change in angle
    input           [127:0] pwm_profile,        // 16 * 8 bit pwm profile 
    
    // PWM Interface    
    input                   pwm_done,           // Updated PWM ratio has been applied (1 cycle long pulse)
    output          [7:0]   pwm_ratio,          // The high-time of the PWM signal out of 255.
    output                  pwm_direction,      // The direction of the motor
    output                  pwm_update,         // Request an update to the PWM ratio
    
    output          [15:0]  debug_signals,

    // I2C Interface
    output                  sck,            // The I2C clock
    inout                   sda             // The I2C bi-directional data
);  

reg     [6:0]   clk_counter;
reg             rd_done_ff, rd_done_ff2;
wire            rd_done;
wire    [11:0]  curr_ang;

// Clock division and crossing
always @(negedge reset_n or posedge clock) begin
    if(~reset_n) begin
        clk_counter[6:0]    <= 7'b0;  
        rd_done_ff          <= 1'b0;
        rd_done_ff2         <= 1'b0;
        current_angle[11:0] <= 12'b0;
    end

    else begin
        clk_counter[6:0]    <= clk_counter[6:0] + 7'b1;
        rd_done_ff          <= rd_done;
        rd_done_ff2         <= rd_done_ff; 

        // When rd_done rises we capture the new value
        if(rd_done_ff & (~rd_done_ff2))
            current_angle[11:0]     <=  curr_ang[11:0];      
    end
end

angle_to_pwm a_to_pwm(
    .reset_n            (reset_n),  	        // Active low reset
    .clock              (clock),	            // The main clock
    .target_angle       (target_angle[11:0]),   // The angle the wheel needs to move to in degrees. This number is multiplied by 2 internally
    .current_angle      (current_angle[11:0]),  // The angle read from the motor encoder
    .pwm_enable         (pwm_enable),           // Enable the PWM
    .pwm_done           (pwm_done),             // Indicator from PWM that the pwm_ratio has been applied
    .angle_update       (angle_update),         // Request to update the angle
    .debug_signals      (debug_signals[15:0]),
    .angle_done         (angle_done),           // Indicator that the angle has been applied 
    .abort_angle        (abort_angle),          // Aborts rotating to angle
    .enable_stall_chk   (enable_stall_chk),     // Enable the stall check
    .consec_chg         (consec_chg[2:0]),      // Number of consecutive changes we want to see before claiming success
    .delay_target       (delay_target[7:0]),    // Number of times to remain on each profile step
    .profile_offset     (profile_offset[7:0]),  // An offset that is added to each of the profile steps
    .cruise_power       (cruise_power[7:0]),    // The amount of power to apply during the cruise phase
    .startup_fail       (startup_fail),         // Error: Motor stalled, unable to startup   .debug_signals  (debug_signals[7:0]),
    .pwm_profile        (pwm_profile[127:0]),   // 16 * 8 bit pwm profile 
    .angle_chg          (angle_chg[63:0]),      // Change in angle
    .pwm_update         (pwm_update),           // Request an update to the PWM ratio
    .pwm_ratio          (pwm_ratio),            // The high-time of the PWM signal out of 255.
    .pwm_direction      (pwm_direction)         // The direction of the motor
);  

i2c i2c(    
    .reset_n            (reset_n),              // Active low reset
    .clock              (clk_counter[6]),       // The main clock
    .angle_done         (angle_done),           // Whether or not we are at the target angle
    .raw_angle          (curr_ang[11:0]),       // The raw angle from the AS5600 
    .rd_done            (rd_done),              // I2C read done pulse           
    .scl                (sck),                  // The I2C clock
    .sda                (sda)                   // The I2C bi-directional data
);

endmodule
