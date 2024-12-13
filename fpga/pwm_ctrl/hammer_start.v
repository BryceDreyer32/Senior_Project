// Copyright 2024
// Bryce's Senior Project
// Description: This module "hammers" the motor to try to get it to start rotating. If it
//              isn't able to get the motor to start, then it throws an error

module hammer_start (
    input               reset_n,        // Active low reset
    input               clock,          // The main clock
    
    input               start_motor,    // Pulse to indicate we want to try to start motor
    input       [11:0]  current_angle,  // The current angle coming from rotation motor
    input               hall_sensor,    // Input from a hall sensor for drive motor
    input               ang_or_drive,   // 0 = angle motor (use current_angle), 1 = drive motor (use hall_sensor)
    input       [2:0]   consec_chg,     // Number of consecutive changes we want to see before claiming success
    input               intend_dir,     // The intended direction of the motor

    input       [3:0]   fwd_count,      // Number of times to apply the forward hammer
    input       [1:0]   rvs_count,      // Number of times to apply the reverse hammer
    input       [3:0]   retry_count,    // Number of retry attempts before admitting defeat

    output reg  [7:0]   pwm_ratio,      // Signal to the PWM
    output reg          pwm_direction,  // Motor direction
    output reg          hammer_done,    // Hammer succeeded
    output reg          error           // Hammer failed    
);

localparam IDLE             = 4'd0;
localparam HAMMER_FORWARD   = 4'd1;
localparam HAMMER_REVERSE   = 4'd2;
localparam HAMMER_PASS      = 4'd3;
localparam HAMMER_FAIL      = 4'd4;

reg     [3:0]   state;
reg     [3:0]   curr_step, ret_cnt;
wire    [7:0]   profile         [15:0];
reg     [11:0]  curr_ang_ff;
reg     [2:0]   chg_cnt;
reg             hall_ff;

// Initialize the hammer profile
assign profile[0][7:0]  = 8'd20;
assign profile[1][7:0]  = 8'd40;
assign profile[2][7:0]  = 8'd80;
assign profile[3][7:0]  = 8'd160;
assign profile[4][7:0]  = 8'd0;
assign profile[5][7:0]  = 8'd80;
assign profile[6][7:0]  = 8'd0;
assign profile[7][7:0]  = 8'd80;
assign profile[8][7:0]  = 8'd0;
assign profile[9][7:0]  = 8'd120;
assign profile[10][7:0] = 8'd0;
assign profile[11][7:0] = 8'd120;
assign profile[12][7:0] = 8'd0;
assign profile[13][7:0] = 8'd160;
assign profile[14][7:0] = 8'd0;
assign profile[15][7:0] = 8'd160;

always @(posedge clock or negedge reset_n) begin
    if(~reset_n) begin
        state           <= IDLE;
        curr_step       <= 4'b0;
        ret_cnt         <= 4'b0;
        curr_ang_ff     <= 12'b0;
        hall_ff         <= 1'b0;
        chg_cnt         <= 3'b0;
        pwm_direction   <= intend_dir;
        hammer_done     <= 1'b0;
        error           <= 1'b0;
    end
    else begin
        if(state == IDLE) begin
            curr_step[3:0]      <= 4'b0;
            ret_cnt[3:0]        <= 4'b0;
            pwm_direction       <= intend_dir;
            hammer_done         <= 1'b0;
            error               <= 1'b0;
            if(start_motor)
                state <= HAMMER_FORWARD;                        
        end

        else if(state == HAMMER_FORWARD) begin
            pwm_direction       <= intend_dir;

            // The following handles how many times we apply each line of the hammer profile
            ret_cnt[2:0]        <= ret_cnt[2:0] + 3'd1;
            if(ret_cnt[2:0] == 3'd7)
                curr_step[3:0]  <= curr_step[3:0] + 4'd1;

            // Apply the PWM power per the profile
            if(curr_step[3:0] < 4'hF) begin
                pwm_ratio[7:0]  <= profile[curr_step][7:0];
            end
            else 
                state   <= HAMMER_REVERSE;

            // Check if we see a change in the hall or angle (depending on config)
            if(~ang_or_drive) begin
                // Angle mode: Look for abs(delta) > 3
                if(curr_ang_ff[11:0] > current_angle[11:0]) begin
                    if((curr_ang_ff[11:0] - current_angle[11:0]) > 12'd3)
                        chg_cnt[2:0]    <= chg_cnt[2:0] + 3'b1;
                end
                else begin
                    if((current_angle[11:0] - curr_ang_ff[11:0]) > 12'd3)
                        chg_cnt[2:0]    <= chg_cnt[2:0] + 3'b1;
                end
            end
            else begin
                // Hall sensor mode: if there's a change then inc the change count
                if(hall_sensor != hall_ff)
                    chg_cnt[2:0]    <= chg_cnt[2:0] + 3'b1;
            end

            // If we've hit the threshold for number of consecutive changes, we can claim success
            if(chg_cnt[2:0] == consec_chg[2:0])
                state   <= HAMMER_PASS;

        end

        else if(state == HAMMER_REVERSE) begin
            pwm_direction       <= ~intend_dir;

            // The following handles how many times we apply each line of the hammer profile
            ret_cnt[2:0]        <= ret_cnt[2:0] + 3'd1;
            if(ret_cnt[2:0] == 3'd7)
                curr_step[3:0]  <= curr_step[3:0] + 4'd1;

            // Apply the PWM power per the profile
            if(curr_step[3:0] < 4'hF) begin
                pwm_ratio[7:0]  <= profile[curr_step][7:0];
            end
            else 
                state   <= HAMMER_FAIL;

            // Check if we see a change in the hall or angle (depending on config)
            if(~ang_or_drive) begin
                // Angle mode: Look for abs(delta) > 3
                if(curr_ang_ff[11:0] > current_angle[11:0]) begin
                    if((curr_ang_ff[11:0] - current_angle[11:0]) > 12'd3)
                        chg_cnt[2:0]    <= chg_cnt[2:0] + 3'b1;
                end
                else begin
                    if((current_angle[11:0] - curr_ang_ff[11:0]) > 12'd3)
                        chg_cnt[2:0]    <= chg_cnt[2:0] + 3'b1;
                end
            end
            else begin
                // Hall sensor mode: if there's a change then inc the change count
                if(hall_sensor != hall_ff)
                    chg_cnt[2:0]    <= chg_cnt[2:0] + 3'b1;
            end

            // If we've hit the threshold for number of consecutive changes, we go back to trying to move in 
            // the direction that we intended
            if(chg_cnt[2:0] == consec_chg[2:0])
                state   <= HAMMER_FORWARD;
        end

        else if(state == HAMMER_FAIL) begin
            error               <= 1'b1;
            state               <= IDLE;
        end

        else if(state == HAMMER_PASS) begin
            hammer_done         <= 1'b1;
            state               <= IDLE;
        end

        curr_ang_ff[11:0]   <= current_angle[11:0];
        hall_ff             <= hall_sensor;
    end
end

endmodule
