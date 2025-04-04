// Copyright 2025
// Bryce's Senior Project
// Description: This RTL is specifically for SparkMax PWM where the following specs hold:
//              Pulse frequency = 131Hz (50-200Hz supported)
//              Full reverse = 1000us
//              Full forward = 2000us
//              Stopped = 1475 - 1525us

module spark_pwm 
(
    input       reset_n,    // Active low reset
    input       clock,      // The main clock
    input       pwm_enable, // Enables the PWM output
    input [7:0] pwm_ratio,  // The high-time of the PWM signal out of 255
    input       pwm_direction, // The motor direction
    input       pwm_update, // Request an update to the PWM ratio
    output reg  pwm_done,   // Updated PWM ratio has been applied (pulse)
    output reg  pwm_signal  // The output PWM wave
);

wire [7:0]  high_time;
reg  [7:0]  pwm_counter;
reg  [7:0]  pwm_target;
reg         pwm_en_sync;

// Calculate the pwm value based upon the incoming value and the the Spark parameters
// @ 131Hz, period = 7.6ms. Input is 255, so each step correlates to 7.6m/255 = 29.8us per step
// So 1ms = 33.5 count, and 2ms = 67 count, so range is ~32-68 with 50 midpoint and +/-18
// Therefore with a 256 input range, this needs to be translated to +/-18, and 256/18 = 14
// which is close to 16. To divide by 16 is same as >> 4 
assign high_time = pwm_direction ?  (8'd50 + (pwm_ratio>>4)):
                                    (8'd50 - (pwm_ratio>>4));

always @( posedge clock or negedge reset_n ) begin
    if( ~reset_n ) begin
        pwm_counter <= 8'h0;
        pwm_target  <= 8'h0;
        pwm_done    <= 1'b0;
        pwm_signal  <= 1'b0;
        pwm_en_sync <= 1'b0;
    end
    else begin
        if(pwm_en_sync) begin
            // Increment the counter (let it roll over)
            pwm_counter <= pwm_counter + 8'h1;

            // If pwm_update is asserted and the pwm_counter is 0, then pull in the pwm_ratio
            if(pwm_counter[7:0] == 8'h0) begin
                if(!pwm_enable)
                    pwm_en_sync <= 1'b0;
                if(pwm_update) begin
                    pwm_target <= high_time;
                    pwm_done   <= 1'b1;
                end                
            end
            // otherwise, if the count < target, then pwm high (if enabled)
            else if(pwm_counter[7:0] < pwm_target[7:0]) begin
                pwm_signal <= 1'b1;
                pwm_done   <= 1'b0;
            end
            // else pwm outputs low
            else begin
                pwm_signal <= 1'b0;
                pwm_done   <= 1'b0;
            end
        end
        else 
            pwm_en_sync <= pwm_enable;
    end
end

endmodule
