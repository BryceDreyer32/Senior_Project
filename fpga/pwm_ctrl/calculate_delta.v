// Copyright 2024
// Bryce's Senior Project
// Description: This module calculates the delta between two positions over multiple clock cycles

module calculate_delta (
    input               reset_n,        // Active low reset
    input               clock,          // The main clock
    input               enable_calc,    // Enable this module to calculate (otherwise output is just 0)
    input       [11:0]  target_angle,   // The angle the wheel needs to move on the 4096 points/rotation scale
    input       [11:0]  current_angle,  // The angle read from the motor encoder
    output  reg         dir_shortest,   // The direction to travel for the shortest path from current -> target
    output  reg [11:0]  delta_angle,    // The shortest distance from current -> target
    output              calc_updated    // Pulse indicating that the calculation has been updated
);

localparam IDLE         = 3'd0;
localparam CALC_DELTA   = 3'd1;
localparam CALC_MIN     = 3'd2;
localparam REPORT       = 3'd3;

reg   [2:0]     ps, ns;
reg   [1:0]     calc_cnt;
reg  [11:0]     calc1, calc2, calc3, calc4;

always @(posedge clock or negedge reset_n) begin
    if(~reset_n) begin
        ps                      <= IDLE;
        calc1[11:0]             <= 12'b0;
        calc2[11:0]             <= 12'b0;
        calc3[11:0]             <= 12'b0;
        calc4[11:0]             <= 12'b0;
        calc_cnt[1:0]           <= 3'b0;
        delta_angle[11:0]       <= 12'b0;
        dir_shortest            <= 1'b0;
    end 
    
    else begin
        ps              <= ns;

        // Perform the calculations for various possible scenarios
        if(ps == CALC_DELTA) begin
            calc1           <= current_angle - target_angle;
            calc2           <= target_angle - current_angle;
            calc3           <= 4096 - target_angle + current_angle;
            calc4           <= 4096 - current_angle + target_angle;
        end
        else if(ps == CALC_MIN) begin
            // Determine the fastest path to the target (and assign to delta_angle), and which direction to rotate
            // Note that we do 1 per clock cycle as attempting to do all in 1 clock could allow glitches to create
            // an incorrect result
            case(calc_cnt[1:0])
                2'b00: begin
                    delta_angle         <= calc1;
                    dir_shortest        <= 1'b1; // CCW
                end
                
                2'b01: begin
                    if(calc2 < delta_angle) begin
                        delta_angle     <= calc2;
                        dir_shortest    <= 1'b0; // CW      
                    end
                end

                2'b10: begin
                    if(calc3 < delta_angle) begin
                        delta_angle     <= calc3;
                        dir_shortest    <= 1'b1; // CCW
                    end
                end

                2'b11: begin
                    if(calc4 < delta_angle) begin
                        delta_angle     <= calc4;
                        dir_shortest    <= 1'b0; // CW
                    end
                end
            endcase
            calc_cnt[1:0] <= calc_cnt[1:0] + 2'b1;
        end
    end
end

assign calc_updated = (ps == REPORT);

always @(*) begin
    case(ps) 
        IDLE: begin
            if(enable_calc)
                ns = CALC_DELTA;
            else
                ns = IDLE;
        end

        CALC_DELTA: begin
            ns = CALC_MIN;
        end 

        CALC_MIN: begin
            if(calc_cnt[1:0] == 2'b11)
                ns = REPORT;
            else
                ns = CALC_MIN;
        end

        REPORT: begin
            ns = IDLE;
        end

    endcase
end


endmodule