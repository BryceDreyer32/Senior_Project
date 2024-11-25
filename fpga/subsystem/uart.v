// Copyright 2024
// Bryce's Senior Project
// Description: This is the RTL for the UART

module uart(
input               reset_n,        // Active low reset
input               clock,          // The main clock
input   [7:0]       tx_data,        // This the 8-bits of data
input   [7:0]       baud_division,  // The division ratio to achieve the desired baud rate
input               tx_start,       // Signal to indicate that the transmission needs to start
output  reg         uart_tx         // UART_TX
);

reg     [7:0]       tx_shadow_reg, baud_counter, wait_count;
reg     [2:0]       ps, ns, bit_count;
reg                 uart_clk; 

localparam      IDLE        = 3'd0;
localparam      START       = 3'd1;
localparam      DATA_TX     = 3'd2;
localparam      PARITY      = 3'd3;
localparam      STOP1       = 3'd4;
localparam      STOP2       = 3'd5;
localparam      PAUSE       = 3'd6;

// Clock Divider for the baud rate
always @(posedge clock or negedge reset_n) begin
    if(~reset_n) begin
        baud_counter[7:0]   <= 8'h0;
        uart_clk            <= 1'h0;
    end

    else begin
        if(baud_counter[7:0] == baud_division[7:0]) begin
            uart_clk            <= ~uart_clk;
            baud_counter[7:0]   <= 8'h0;
        end

        else 
            baud_counter[7:0]   <= baud_counter[7:0] + 8'h1;
    end
end

// FSM to control the UART
always @(posedge uart_clk or negedge reset_n)  begin
    if(~reset_n) begin
        ps                  <= IDLE;
        uart_tx             <= 1'b1;
        tx_shadow_reg[7:0]  <= 8'b0;
        bit_count[2:0]      <= 3'b0;
        wait_count[7:0]     <= 8'b0;
    end

    else begin
        ps <= ns;
        if(ps == START) begin
            tx_shadow_reg[7:0]      <= tx_data[7:0];
            uart_tx                 <= 1'b0;
        end

        else if(ps == DATA_TX) begin
            uart_tx         <= tx_shadow_reg[bit_count];
            bit_count[2:0]  <= bit_count[2:0] + 3'b1;
        end

        else if(ps == PARITY)
            uart_tx     <= ^(tx_shadow_reg[7:0]);

        
        else if(ps == PAUSE) begin
            wait_count[7:0] <= wait_count[7:0] + 8'b1;
            uart_tx         <= 1'b1;
        end
        
        // uart_tx default is 1 unless a start, data , parity is being set
        else begin
            bit_count[2:0]  <= 3'b0;
            uart_tx         <= 1'b1;
            wait_count[7:0] <= 8'b0;
        end
    end
end

always @(*) begin

    case(ps)
        IDLE: begin
            if(tx_start)
                ns = START;
            else
                ns = IDLE;
        end

        START: begin
            ns = DATA_TX;
        end

        DATA_TX: begin
            if(bit_count[2:0] == 3'b111)
                ns = PARITY;
            else
                ns = DATA_TX;
        end

        PARITY: begin
            ns = STOP1;
        end

        STOP1: begin
            ns = STOP2;
        end

        STOP2: begin
            ns = PAUSE;
        end

        PAUSE: begin
            if(wait_count[7:0] == 8'hF)
                ns = IDLE;
            else
                ns = PAUSE;
        end

        default: begin
            ns = IDLE;
        end
    endcase
end

endmodule

