// Copyright 2024
// Bryce's Senior Project
// Description: This is the RTL for PWM Control

module pwm_ctrl(
    input                   reset_n,        // Active low reset
    input                   clock,          // The main clock

    //FPGA Subsystem Interface
    input           [11:0]  target_angle,   // The angle the wheel needs to move to in degrees. This number is multiplied by 2 internally
    input                   angle_update,   // Signals when an angle update is available
    output                  angle_done,     // Output sent when angle has been adjusted to target_angle
    output  reg     [11:0]  current_angle,  // Angle we are currently at from I2C

    //PWM Interface
    input                   pwm_done,       // Updated PWM ratio has been applied (1 cycle long pulse)
    output                  pwm_enable,     // Enables the PWM output
    output          [7:0]   pwm_ratio,      // The high-time of the PWM signal out of 255.
    output                  pwm_update,     // Request an update to the PWM ratio

    output          [7:0]   debug_signals,

    //I2C Interface
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
    .reset_n        (reset_n),  	        // Active low reset
    .clock          (clock),	            // The main clock
    .target_angle   (target_angle[11:0]),   // The angle the wheel needs to move to in degrees. This number is multiplied by 2 internally
    .current_angle  (current_angle[11:0]),  // The angle read from the motor encoder
    .pwm_done       (pwm_done),             // Indicator from PWM that the pwm_ratio has been applied
    .angle_update   (angle_update),         // Request to update the angle
    .angle_done     (angle_done),           // Indicator that the angle has been applied 
    .debug_signals  (debug_signals[7:0]),
    .pwm_enable     (pwm_enable),   
    .pwm_update     (pwm_update),           // Request an update to the PWM ratio
    .pwm_ratio      (pwm_ratio[7:0]),       // The high-time of the PWM signal out of 255.
    .pwm_direction  ()                      // The direction of the motor
);  

i2c i2c(    
    .reset_n        (reset_n),              // Active low reset
    .clock          (clk_counter[6]),       // The main clock
    .angle_done     (angle_done),           // Whether or not we are at the target angle
    .raw_angle      (curr_ang[11:0]),       // The raw angle from the AS5600 
    .rd_done        (rd_done),              // I2C read done pulse           
    .scl            (sck),                  // The I2C clock
    .sda            (sda)                   // The I2C bi-directional data
);

endmodule
