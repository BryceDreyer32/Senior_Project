// Copyright 2024
// Bryce's Senior Project
// Description: This is the RTL for the I2C which communicates between the PWM controller and the encoder.
module top(
    input               clock,          // The main clock          
    output              scl,            // The I2C clock
    inout               sda             // The I2C bi-directional data
);

reg     [5:0]   clk_counter;
always @(posedge clock)
    clk_counter[5:0]    <= clk_counter[5:0] + 6'b1;

i2c dut(
    .clock          (clk_counter[5]),
    .reset_n        (1'b1),
    .angle_done     (1'b0),
    .raw_angle      (),
    .scl            (scl),
    .sda            (sda)
);
endmodule