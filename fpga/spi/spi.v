// Copyright 2024
// Bryce's Senior Project
// Description: This is the RTL for SPI

module spi ( 
    input                   reset_n,    // Active low reset
    input                   clock,      // The main clock
    input                   spi_clk,    // The SPI clock
    input                   cs_n,       // Active low chip select
    input                   mosi,       // Master out slave in
    input           [7:0]   rd_data,

    output                  miso,       // Master in slave out (SPI mode 0)
    output reg              miso_clocked,
    output reg      [5:0]   address,
    output reg              write_en,
    output reg      [7:0]   wr_data,
    output reg              read_en
);

wire                shift_en;
wire                rx_parity, tx_parity;
reg                 csn_wenthigh, csn_wenthigh_ff, csn_wentlow;
reg                 read, write, load_miso;
reg                 parity;
reg                 cs_n_ff, cs_n_ff2;
reg     [15:0]      shift_in_reg, shift_out_reg, mosi_reg, miso_reg;

// The shift_en is the clock masked by cs_n
assign shift_en = spi_clk & (~cs_n);

// rx_parity is the xor of all the mosi_reg bits (excluding bit 14, which is the parity bit itself)
assign rx_parity = ^({mosi_reg[15], mosi_reg[13:0]});

// assign mosi_reg vectors to more meaningful names
always @(posedge clock or negedge reset_n) begin
    if(~reset_n) begin
        wr_data[7:0]    <= 8'b0;
        address[5:0]    <= 6'b0;
        parity          <= 1'b0;
        read            <= 1'b0;
        write           <= 1'b0;
    end

    else begin
        if(csn_wenthigh) begin
            wr_data[7:0]    <= mosi_reg[7:0];
            address[5:0]    <= mosi_reg[13:8];
            parity          <= mosi_reg[14];
            read            <= mosi_reg[15];
            write           <= ~mosi_reg[15];
        end
    end   
end

// data_valid asserts when the recived parity matches the calculated parity and mosi_reg is not all 1s
assign data_valid = (mosi_reg[15:0] != 16'hFF) & ~(parity ^ rx_parity);

// Synchronize the cs_n from the spi_clk domain to the FPGA clock domain
always @(posedge clock or negedge reset_n) begin
    if(~reset_n) begin
        cs_n_ff             <= 1'b0;
        cs_n_ff2            <= 1'b0;
        csn_wenthigh        <= 1'b0;
        csn_wenthigh_ff     <= 1'b0;
        csn_wentlow         <= 1'b0;
    end

    else begin
        cs_n_ff         <= cs_n;
        cs_n_ff2        <= cs_n_ff;
        // csn_wenthigh is an indicator that the csn was low in the previous cycle, but has now gone high
        // csn_wentlow is the opposite 
        csn_wenthigh        <= cs_n_ff & (~cs_n_ff2);
        csn_wenthigh_ff     <= csn_wenthigh;
        csn_wentlow         <= (~cs_n_ff) & cs_n_ff2; 

    end
end

// Compute the tx_parity (1 for read, address, and data) 
assign tx_parity    =   ^({1'b1, address[5:0], rd_data[7:0]});

// ----------------------------------- READ / WRITE ENABLE ------------------------------------
always @(posedge clock or negedge reset_n) begin
    if(~reset_n) begin
        write_en    <= 1'b0;
        read_en     <= 1'b0;
    end    
    else begin
        write_en    <= csn_wenthigh_ff & write & data_valid;
        read_en     <= csn_wenthigh_ff & read & data_valid; 
    end
end


// -------------------------------------- MOSI REGISTERS --------------------------------------
// The shift_in_reg gets filled MSB to LSB with MOSI data
always @(posedge shift_en or negedge reset_n) begin
    if(~reset_n)
        shift_in_reg[15:0] <= 16'h0;
    else begin
        shift_in_reg [15:1] <= shift_in_reg[14:0];
        shift_in_reg[0] <= mosi;
    end
end

// The mosi_reg is updated from the shift_in_reg when csn_wenthigh is asserted
always @(posedge csn_wenthigh or negedge reset_n) begin
    if(~reset_n)
        mosi_reg[15:0]  <= 16'h0;
    else 
        mosi_reg [15:0] <= shift_in_reg[15:0];
end

// -------------------------------------- MISO REGISTERS --------------------------------------
/*always @(posedge clock or negedge reset_n) begin
    if(~reset_n) begin
        miso_reg[15:0]      <= 16'b0;
    end

    else begin
        miso_reg[7:0]       <= rd_data[7:0];
        miso_reg[8]         <= tx_parity;
        miso_reg[15:9]      <= 7'b0;
    end
end

always @(posedge clock or negedge reset_n) begin
    if(~reset_n)
        shift_out_reg[15:0]     <= 16'b0;
    else 
        if(csn_wentlow)
            shift_out_reg[15:0]     <= miso_reg[15:0];
    
end

always @(posedge ~shift_en *or posedge csn_wentlow*) begin
//    if(csn_wentlow)
//        shift_out_reg[15:0]     <= miso_reg[15:0];
//    else 
        shift_out_reg[15:0]     <= (shift_out_reg[15:0] << 1);
end
*/

always @(posedge csn_wentlow or negedge reset_n) begin
    if(~reset_n) begin
        shift_out_reg[15:0]      <= 16'b0;
    end

    else begin
        shift_out_reg[15:0] <= 16'hDEAD;
/*        shift_out_reg[7:0]       <= rd_data[7:0];
        shift_out_reg[8]         <= tx_parity;
        shift_out_reg[15:9]      <= 7'b0;*/
    end
end

always @(negedge shift_en)
    shift_out_reg[15:0]     <= (shift_out_reg[15:0] << 1);

assign miso     = shift_out_reg[15]/* | cs_n*/;

always @(posedge clock)
    miso_clocked <= miso;

endmodule