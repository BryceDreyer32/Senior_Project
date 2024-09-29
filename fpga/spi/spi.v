// Copyright 2024
// Bryce's Senior Project
// Description: This is the RTL for SPI

module spi ( 
    input           reset_n,    // Active low reset
    input           clock,      // The main clock
    input           cs_n,       // Active low chip select
    input           mosi,       // Master out slave in

    output reg [15:0] shadow_reg  //The output of the SPI
);

wire shift_en;
reg [15:0] shift_reg;

// The shift_en is the clock masked by cs_n
assign shift_en = clock & (~cs_n);

// The shift_reg gets filled MSB to LSB with MOSI data
always @(posedge shift_en or negedge reset_n) begin
    if(~reset_n)
        shift_reg[15:0] <= 16'h0;
    else begin
        shift_reg [15:1] <= shift_reg[14:0];
        shift_reg[0] <= mosi;
    end
end

// The shadow_reg is updated from the shift_reg when cs_n is asserted
always @(posedge cs_n or negedge reset_n) begin
    if(~reset_n)
        shadow_reg[15:0] <= 16'h0;
    else 
        shadow_reg [15:0] <= shift_reg[15:0];
end
endmodule