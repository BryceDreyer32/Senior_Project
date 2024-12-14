// Copyright 2024
// Bryce's Senior Project
// Description: This is the RTL for PWM
module pwm 
(
    input       reset_n,    // Active low reset
    input       clock,      // The main clock
    input       pwm_enable, // Enables the PWM output
    input [7:0] pwm_ratio,  // The high-time of the PWM signal out of 255.
    input       pwm_update, // Request an update to the PWM ratio
    output reg  pwm_done,   // Updated PWM ratio has been applied (pulse)
    output      pwm_signal  // The output PWM wave
);

reg [7:0]   pwm_counter;
reg [7:0]   pwm_target;

always @( posedge clock or negedge reset_n ) begin
    if( ~reset_n ) begin
        pwm_counter <= 8'h0;
        pwm_target  <= 8'h0;
        pwm_done    <= 1'b0;
    end
    else begin
        // Increment the counter (let it roll over)
        pwm_counter <= pwm_counter + 8'h1;

        // If pwm_update is asserted and the pwm_counter is 0, then pull in the pwm_ratio
        if( pwm_update & (pwm_counter[7:0] == 8'h0) ) begin
            pwm_target <= pwm_ratio;
            pwm_done   <= 1'b1;
        end
        else
            pwm_done   <= 1'b0;
    end
end

// Output pwm_signal is 1 when pwm_enable is set and the pwm_counter <= pwm_target
assign pwm_signal = pwm_enable & (pwm_counter[7:0] < pwm_target[7:0]);

endmodule
