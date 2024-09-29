// Copyright 2024
// Bryce's Senior Project
// Description: This is the RTL for the I2C which communicates between the PWM controller and the encoder.
module i2c_old(
    input               reset_n,        // Active low reset
    input               clock,          // The main clock
    input               get_data,       // A pulse to tell I2C to pull data
    output reg  [11:0]  raw_angle,      // The raw angle from the AS5600            
    output reg          sck,            // The I2C clock
    inout               sda             // The I2C bi-directional data
);

localparam      ENCODER_ADDRESS     = 7'h36;
localparam      CLOCK_DIV_RATIO     = 6'h32;

reg             sck_local;
reg     [5:0]   sck_counter;

// Clock division 
always @(posedge clock or negedge reset_n) begin
    if(~reset_n) begin
        sck_local   <= 1'b0;
        sck_counter <= 6'b0;
    end
    else begin
        if(sck_counter == CLOCK_DIV_RATIO / 2) begin
            sck_counter <= 6'b0;
            sck_local   <= ~sck_local;
        end
        else
            sck_counter <= sck_counter + 6'b1;
    end
end


endmodule