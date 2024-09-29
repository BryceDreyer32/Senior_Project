// Copyright 2024
// Bryce's Senior Project
// Description: This is the TestBench for SPI

module test();
  reg clock = 0;
  reg reset_n = 0;
  reg cs_n = 1;
  wire mosi;
  wire [15:0] shadow_reg;
  wire gated_clock;
  reg [15:0] mosi_reg;

 always
    #1  clock = ~clock;
assign gated_clock = clock & ~cs_n;

initial begin
    $display("Starting SPI test");
    
    $display("Deasserting reset");
    #10 reset_n = 1;

    $display("Asserting cs_n");
    #2 cs_n = 0;

    $display("Running for 16 cycles");
    #32;

    $display("Deasserting cs_n");
    cs_n = 1;

    #10 $finish;
end

always @(negedge gated_clock or negedge reset_n) begin
    if(~reset_n)
        mosi_reg[15:0] <= 16'hDEAD;
    else
        mosi_reg[15:1] <= mosi_reg[14:0];
end

assign mosi = mosi_reg[15];

    spi dut(
        .reset_n        (reset_n),
        .clock          (clock),
        .cs_n           (cs_n),
        .mosi           (mosi),
        .shadow_reg     (shadow_reg)
    );

initial begin
    $dumpfile("spi.vcd");
    $dumpvars(0,test);
  end

endmodule