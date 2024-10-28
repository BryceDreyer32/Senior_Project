// Copyright 2024
// Bryce's Senior Project
// Description: This is the TestBench for SPI

module test();
  reg clock = 0;
  reg reset_n = 0;
  reg angle_done = 1;
  wire   [11:0] raw_angle;
  wire scl, sda;
  reg sda_temp;

 always
    #1  clock = ~clock;

initial begin
    $display("Starting i2c test");
    sda_temp = 1'bZ;
    
    $display("Deasserting reset");
    #10 reset_n = 1;

    $display("Setting angle_done to 0");
    #10 angle_done = 0;

    #150;
    /* #8 sda_temp = 1'b1;
    #8 sda_temp = 1'b1;
    #8 sda_temp = 1'b0;
    #8 sda_temp = 1'b0;
    #8 sda_temp = 1'b1;
    #8 sda_temp = 1'b0;
    #8 sda_temp = 1'b1;
    #8 sda_temp = 1'b0;
    #8 sda_temp = 1'bZ; */

    #500;

    #10 $finish;
end

assign sda = sda_temp;

i2c dut(
    .clock          (clock),
    .reset_n        (reset_n),
    .angle_done     (angle_done),
    .raw_angle      (raw_angle[11:0]),
    .scl            (scl),
    .sda            (sda)
);

initial begin
    $dumpfile("i2c.vcd");
    $dumpvars(0,test);
  end

endmodule