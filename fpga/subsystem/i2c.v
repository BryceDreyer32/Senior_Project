// Copyright 2024
// Bryce's Senior Project
// Description: This is the RTL for the I2C (adapted from Lushay Labs)
module i2c (
    input               clk,
    input               reset_n,
    input               sdaIn,
    output reg          sdaOutReg,
    output reg          isSending,
    output reg          scl,
    input [1:0]         instruction,
    input               enable,
    input [7:0]         byteToSend,
    input               send_nack,
    output reg [7:0]    byteReceived,
    output reg          complete
);

    localparam INST_START_TX = 0;
    localparam INST_STOP_TX = 1;
    localparam INST_READ_BYTE = 2;
    localparam INST_WRITE_BYTE = 3;
    localparam STATE_IDLE = 4;
    localparam STATE_DONE = 5;
    localparam STATE_SEND_ACK = 6;
    localparam STATE_RCV_ACK = 7;
    localparam STATE_SEND_NACK = 8;

    reg [6:0] clockDivider = 0;

    reg [2:0] ps, ns;
    reg [2:0] bitToSend = 0;

    always @(posedge clk or negedge reset_n) begin
        if(~reset_n) begin
            ps              <= STATE_IDLE;
            complete        <= 1'b0;
            clockDivider    <= 7'b0;
            bitToSend       <= 3'b0;
            isSending       <= 1'b0; 
            byteReceived    <= 8'b0;
        end
        else begin
            ps              <= ns;
            if((ps == STATE_IDLE) & enable) begin
                complete        <= 1'b0;
                clockDivider    <= 7'b0;
                bitToSend       <= 3'b0; 
            end
            else if(ps == INST_START_TX) begin
                isSending       <= 1'b1;
                complete        <= 1'b0;
                byteReceived    <= 8'b0;
                clockDivider    <= clockDivider + 7'b1;
                if (clockDivider[6:5] == 2'b00) begin
                    scl         <= 1'b1;
                    sdaOutReg   <= 1'b1;
                end else if (clockDivider[6:5] == 2'b01) begin
                    sdaOutReg   <= 1'b0;
                end else if (clockDivider[6:5] == 2'b10) begin
                    scl         <= 1'b0;
                end   
            end
            else if(ps == INST_STOP_TX) begin
                isSending       <= 1'b1;
                clockDivider    <= clockDivider + 7'b1;
                if (clockDivider[6:5] == 2'b00) begin
                    scl         <= 1'b0;
                    sdaOutReg   <= 1'b0;
                end else if (clockDivider[6:5] == 2'b01) begin
                    scl         <= 1'b1;
                end else if (clockDivider[6:5] == 2'b10) begin
                    sdaOutReg   <= 1'b1;
                end
            end
            else if(ps == INST_READ_BYTE) begin
                isSending       <= 1'b0;
                clockDivider    <= clockDivider + 7'b1;
                if (clockDivider[6:5] == 2'b00) begin
                    scl         <= 1'b0;
                end else if (clockDivider[6:5] == 2'b01) begin
                    scl         <= 1'b1;
                end else if (clockDivider == 7'b1000000) begin
                    byteReceived <= {byteReceived[6:0], sdaIn ? 1'b1 : 1'b0};
                end else if (clockDivider == 7'b1111111) begin
                    bitToSend   <= bitToSend + 3'b1;
                end else if (clockDivider[6:5] == 2'b11) begin
                    scl         <= 1'b0;
                end
            end
            else if(ps == STATE_SEND_ACK) begin
                isSending       <= 1'b1;
                sdaOutReg       <= 1'b0;
                clockDivider    <= clockDivider + 7'b1;
                if (clockDivider[6:5] == 2'b01) begin
                    scl         <= 1'b1;
                end else if (clockDivider[6:5] == 2'b11) begin
                    scl         <= 1'b0;
                end
            end
            else if(ps == INST_WRITE_BYTE) begin
                isSending       <= 1'b1;
                clockDivider    <= clockDivider + 7'b1;
                sdaOutReg       <= byteToSend[3'd7-bitToSend] ? 1'b1 : 1'b0;

                if (clockDivider[6:5] == 2'b00) begin
                    scl         <= 1'b0;
                end else if (clockDivider[6:5] == 2'b01) begin
                    scl         <= 1'b1;
                end else if (clockDivider == 7'b1111111) begin
                    bitToSend   <= bitToSend + 3'b1;
                end else if (clockDivider[6:5] == 2'b11) begin
                    scl         <= 1'b0;
                end
            end
            else if(ps == STATE_RCV_ACK) begin
                isSending       <= 1'b0;
                clockDivider    <= clockDivider + 7'b1;

                if (clockDivider[6:5] == 2'b01) begin
                    scl         <= 1'b1;
                end else if (clockDivider[6:5] == 2'b11) begin
                    scl         <= 1'b0;
                end
            end
            else if(ps == STATE_SEND_NACK) begin
                isSending       <= 1'b1;
                sdaOutReg       <= 1'b1;
                clockDivider    <= clockDivider + 7'b1;
                if (clockDivider[6:5] == 2'b01) begin
                    scl         <= 1'b1;
                end else if (clockDivider[6:5] == 2'b11) begin
                    scl         <= 1'b0;
                end    
            end
            else if(ps == STATE_DONE) begin
                complete        <= 1'b1;
            end
        end
    end

    always @(*) begin
        case (ps)
            STATE_IDLE: begin
                if(send_nack)
                    ns = STATE_SEND_NACK;
                else
                    ns = {1'b0,instruction};
            end

            INST_START_TX: begin
                if (clockDivider[6:5] == 2'b11) 
                    ns = STATE_DONE;
                else
                    ns = INST_START_TX;   
            end 
            
            INST_STOP_TX: begin
                if (clockDivider[6:5] == 2'b11) 
                    ns = STATE_DONE;
                else
                    ns = INST_STOP_TX;
            end
            
            INST_READ_BYTE: begin
                if (clockDivider == 7'b1111111) begin
                    if (bitToSend == 3'b111) begin
                        ns = STATE_SEND_ACK;
                    end
                    else
                        ns = INST_READ_BYTE;
                end
                else   
                    ns = INST_READ_BYTE;
            end

            STATE_SEND_ACK: begin
                if (clockDivider == 7'b1111111) 
                    ns = STATE_DONE;
                else 
                    ns = STATE_SEND_ACK;
            end
            
            INST_WRITE_BYTE: begin
                 if (clockDivider == 7'b1111111) begin
                    if (bitToSend == 3'b111) 
                        ns = STATE_RCV_ACK;
                    else
                        ns = INST_WRITE_BYTE;
                 end
                 else 
                    ns = INST_WRITE_BYTE;
            end
            
            STATE_RCV_ACK: begin
                if(clockDivider == 7'b1111111) 
                    ns = STATE_DONE;
                else
                    ns = STATE_RCV_ACK;
            end

            STATE_SEND_NACK: begin
                if (clockDivider == 7'b1111111)
                    ns = STATE_DONE;
                else
                    ns = STATE_SEND_NACK;    
            end

            STATE_DONE: begin
                if (~enable)
                    ns = STATE_IDLE;
                else
                    ns = STATE_DONE;
            end

            default: begin
                ns = STATE_IDLE;
            end 
        endcase
    end
endmodule