// Copyright 2024
// Bryce's Senior Project
// Description: This is the RTL for PWM Control

module pwm_ctrl(
    input               reset_n,        // Active low reset
    input               clock,          // The main clock

    //FPGA Subsystem Interface
    input   [11:0]      target_angle,   // The angle the wheel needs to move to in degrees. This number is multiplied by 2 internally
    input               angle_update,   // Signals when an angle update is available
    output              angle_done,     // Output sent when angle has been adjusted to target_angle
    output  [11:0]      current_angle,  // Angle we are currently at from I2C

    //PWM Interface
    input               pwm_done,       // Updated PWM ratio has been applied (1 cycle long pulse)
    output              pwm_enable,     // Enables the PWM output
    output  [7:0]       pwm_ratio,      // The high-time of the PWM signal out of 255.
    output              pwm_update,     // Request an update to the PWM ratio

    //I2C Interface
    output              sck,            // The I2C clock
    inout               sda             // The I2C bi-directional data
);  

angle_to_pwm a_to_pwm(
    .reset_n        (reset_n),  	        // Active low reset
    .clock          (clock),	            // The main clock
    .target_angle   (target_angle[11:0]),   // The angle the wheel needs to move to in degrees. This number is multiplied by 2 internally
    .current_angle  (current_angle[11:0]),  // The angle read from the motor encoder
    .pwm_done       (pwm_done),             // Indicator from PWM that the pwm_ratio has been applied
    .angle_update   (angle_update),         // Request to update the angle
    .angle_done     (angle_done),           // Indicator that the angle has been applied 
    .pwm_enable     (pwm_enable),   
    .pwm_update     (pwm_update),           // Request an update to the PWM ratio
    .pwm_ratio      (pwm_ratio),            // The high-time of the PWM signal out of 255.
    .pwm_direction  ()                      // The direction of the motor
);  

i2c i2c(    
    .reset_n        (reset_n),              // Active low reset
    .clock          (clock),                // The main clock
    .angle_done     (angle_done),           // Whether or not we are at the target angle
    .raw_angle      (current_angle[11:0]),  // The raw angle from the AS5600            
    .scl            (sck),                  // The I2C clock
    .sda            (sda)                   // The I2C bi-directional data
);

endmodule