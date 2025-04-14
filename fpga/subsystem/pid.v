// Copyright 2025
// Bryce's Senior Project
// Description: This is the Proportional-Integral-Derivative (PID) controller for the motor

module pid(
    input               reset_n,            // Active low reset
    input               clock,              // The main clock
    input       [11:0]  target_angle,       // The angle the wheel needs to move on the 4096 points/rotation scale
    input       [11:0]  current_angle,      // The angle read from the motor encoder
    input               pwm_enable,         // PWM enable
    input               pwm_done,           // Indicator from PWM that the pwm_ratio has been applied
    input               angle_update,       // Request to update the angle
    input               abort_angle,        // Aborts rotating to angle

    input               enable_stall_chk,   // Enable the stall check
    input       [7:0]   delay_target,       // Number of times to remain on each profile step
    output reg          startup_fail,       // Error: Motor stalled, unable to startup
    input       [7:0]   kp,                 // Proportional Constant: fixed point 4.4
    input       [3:0]   ki,                 // Integral Constant: fixed point 0.4
    input       [3:0]   kd,                 // Derivative Constant: fixed point 0.4

    output      [15:0]  debug_signals,  
    output reg          angle_done,         // Indicator that the angle has been applied 
    output reg          pwm_update,         // Request an update to the PWM ratio
    output reg  [7:0]   pwm_ratio,          // The high-time of the PWM signal out of 255.
    output reg          pwm_direction       // The direction of the motor
);

wire    [31:0]      proportional_error;             // The proportional instantenous error
wire    [31:0]      integral_error;                 // The cumulative error
wire    [31:0]      derivative_error;               // The derivative error
wire    [15:0]      elapsed_time;                   // The elapsed time since the last update
wire    [15:0]      delta_12p4, last_delta_12p4;    // The last error from the PID controller
wire    [15:0]      target_12p4, current_12p4;      // Fixed point 12.4 format

assign target_12p4  = target_angle  << 4;
assign current_12p4 = current_angle << 4;
assign delta_12p4   = delta_angle   << 4;

always @(negedge reset_n or posedge clock) begin
    if(~reset_n) begin
        angle_done          <= 1'b0; 
        pwm_update          <= 1'b0; 
        pwm_ratio           <= 8'b0; 
        pwm_direction       <= 1'b0; 
        startup_fail        <= 1'b0; 
        debug_signals       <= 16'hDEAD;
        proportional_error  <= 32'b0;
        integral_error      <= 32'b0;
        derivative_error    <= 32'b0;
        elapsed_time        <= 16'b0;
        last_delta_12p4     <= 12'b0;
    end
    else begin
        if (pwm_enable) begin
            elapsed_time            <= elapsed_time + 16'b1;
            last_delta_12p4         <= delta_12p4;
            proportional_error      <= kp * delta_12p4;
            integral_error          <= ki * delta_12p4 * elapsed_time;
            derivative_error        <= kd * (delta_12p4 - last_delta_12p4) / elapsed_time;
            pwm_ratio               <= (proportional_error + integral_error + derivative_error) >> 4;
        end 
        else begin
            elapsed_time    <= 16'b0;
            pwm_ratio       <= 8'b0;
        end
    end
end

calculate_delta calc (
    .reset_n        (reset_n),      
    .clock          (clock),        
    .enable_calc    (enable_calc),  
    .target_angle   (target_angle[11:0]), 
    .current_angle  (current_angle[11:0]),
    .dir_shortest   (pwm_direction), 
    .delta_angle    (delta_angle[11:0]),  
    .calc_updated   (calc_updated)
);
endmodule