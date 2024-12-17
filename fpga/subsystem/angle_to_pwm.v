// Copyright 2024
// Bryce's Senior Project
// Description: This is the RTL for "Angle to PWM". This will include calculating the PWM control values based on current and target angles
//              and includes the acceleration and deceleration profiles. 

module angle_to_pwm(
    input               reset_n,        // Active low reset
    input               clock,          // The main clock
    input       [11:0]  target_angle,   // The angle the wheel needs to move on the 4096 points/rotation scale
    input       [11:0]  current_angle,  // The angle read from the motor encoder
    input               pwm_done,       // Indicator from PWM that the pwm_ratio has been applied
    input               angle_update,   // Request to update the angle
    input               abort_angle,    // Aborts rotating to angle

    input               enable_hammer,  // Enables hammer acceleration (vs linear)
    input       [3:0]   fwd_count,      // Number of times to apply the forward hammer
    input       [3:0]   rvs_count,      // Number of times to apply the reverse hammer
    input       [1:0]   retry_count,    // Number of retry attempts before admitting defeat
    input       [2:0]   consec_chg,     // Number of consecutive changes we want to see before claiming success
    input       [7:0]   delay_target,   // Number of times to remain on each profile step
    output reg          startup_fail,   // Error: Motor stalled, unable to startup

    output      [15:0]  debug_signals,
    output reg          angle_done,     // Indicator that the angle has been applied 
    output reg          pwm_enable,     // PWM enable
    output reg          pwm_update,     // Request an update to the PWM ratio
    output reg  [7:0]   pwm_ratio,      // The high-time of the PWM signal out of 255.
    output reg          pwm_direction   // The direction of the motor
);

// States
localparam IDLE             = 4'd0;
localparam CALC             = 4'd1;
localparam ACCEL            = 4'd2;
localparam HAMMER_FORWARD   = 4'd3;
localparam HAMMER_REVERSE   = 4'd4;
localparam HAMMER_PASS      = 4'd5;
localparam HAMMER_FAIL      = 4'd6;
localparam CRUISE           = 4'd7;
localparam DECEL            = 4'd8; 
localparam SHUTDOWN         = 4'd9; 

localparam SMALL_DELTA  = 4'd8;
localparam MED_DELTA    = 4'd10;
localparam BIG_DELTA    = 4'd14;

localparam TARGET_TOLERANCE     = 12'd5; 

reg   [3:0] ps, ns;
wire  [7:0] hammer_profile               [15:0];
wire  [7:0] linear_profile               [15:0];
wire [11:0] delta_angle;
reg  [11:0] curr_ang_ff;
reg   [3:0] num_steps;
reg   [3:0] curr_step;
reg   [7:0] profile_delay;
wire        calc_updated;
wire        dir_shortest;
reg         enable_calc;
reg         pwm_done_ff, pwm_done_went_high;
reg   [1:0] retry_cnt;
reg   [2:0] chg_cnt;
reg         hammer_chk;
reg         run_stall;

assign debug_signals = {startup_fail, run_stall, retry_cnt[1:0], pwm_direction, angle_update, abort_angle, pwm_done,
                        chg_cnt[2:0], pwm_update, ps[3:0]};

// Initialize the acceleration and deceleration profiles
assign hammer_profile[0][7:0]  = 8'd40;
assign hammer_profile[1][7:0]  = 8'd80;
assign hammer_profile[2][7:0]  = 8'd140;
assign hammer_profile[3][7:0]  = 8'd0;
assign hammer_profile[4][7:0]  = 8'd100;
assign hammer_profile[5][7:0]  = 8'd0;
assign hammer_profile[6][7:0]  = 8'd120;
assign hammer_profile[7][7:0]  = 8'd0;
assign hammer_profile[8][7:0]  = 8'd140;
assign hammer_profile[9][7:0]  = 8'd0;
assign hammer_profile[10][7:0] = 8'd160;
assign hammer_profile[11][7:0] = 8'd0;
assign hammer_profile[12][7:0] = 8'd180;
assign hammer_profile[13][7:0] = 8'd0;
assign hammer_profile[14][7:0] = 8'd200;
assign hammer_profile[15][7:0] = 8'd0;

assign linear_profile[0][7:0]  = 8'd11;
assign linear_profile[1][7:0]  = 8'd29;
assign linear_profile[2][7:0]  = 8'd48;
assign linear_profile[3][7:0]  = 8'd65;
assign linear_profile[4][7:0]  = 8'd82;
assign linear_profile[5][7:0]  = 8'd99;
assign linear_profile[6][7:0]  = 8'd113;
assign linear_profile[7][7:0]  = 8'd128;
assign linear_profile[8][7:0]  = 8'd141;
assign linear_profile[9][7:0]  = 8'd153;
assign linear_profile[10][7:0] = 8'd164;
assign linear_profile[11][7:0] = 8'd174;
assign linear_profile[12][7:0] = 8'd185;
assign linear_profile[13][7:0] = 8'd193;
assign linear_profile[14][7:0] = 8'd200;
assign linear_profile[15][7:0] = 8'd206;


always @(negedge reset_n or posedge clock)
    if(~reset_n) begin
        ps                  <= IDLE;
        curr_step[3:0]      <= 4'b0;
        curr_ang_ff[11:0]   <= 12'b0;
        pwm_ratio[7:0]      <= 8'd0;
        pwm_enable          <= 1'b1;
        pwm_update          <= 1'b0;
        pwm_done_ff         <= 1'b0;
        pwm_done_went_high  <= 1'b0;
        profile_delay[7:0]  <= 8'b0;
        angle_done          <= 1'b0;
        enable_calc         <= 1'b0;
        num_steps[3:0]      <= MED_DELTA;
        pwm_direction       <= 1'b0;
        retry_cnt           <= 2'b0;
        hammer_chk          <= 1'b0;
        chg_cnt[2:0]        <= 3'b0;
        startup_fail        <= 1'b0;
        run_stall           <= 1'b0;
    end
    else begin
        ps                  <= ns;
        pwm_enable          <= 1'b1;

        // The pwm_done_went_high ensures that the PWM has had reset this signal
        // and it is now going high. This avoids a race condition where the ratio
        // is updated, and the pwm_done is immediately read: it may not have had 
        // time to reset, and therefore gives a false impression the new ratio
        // has been absorbed when in reality it hasn't.
        pwm_done_ff         <= pwm_done_ff;
        pwm_done_went_high  <= ~pwm_done_ff & pwm_done;

        // Indicator that asserts at end of hammer profile 
        hammer_chk      <= 1'b0;

        if(ps == IDLE) begin
            // If we are in IDLE force the ratio to 0
            curr_step[3:0]  <= 4'b0;
            pwm_ratio[7:0]  <= 8'd0;
            pwm_direction   <= dir_shortest;
            pwm_update      <= 1'b0;
            enable_calc     <= 1'b0;
            chg_cnt[2:0]    <= 3'b0;
            hammer_chk      <= 1'b0;
        end

        else if( ps == CALC) begin
            curr_step[3:0]  <= 4'b0;
            pwm_ratio[7:0]  <= 8'd0;
            pwm_direction   <= dir_shortest;
            enable_calc     <= 1'b1;
            retry_cnt[1:0]  <= 2'b0;
            startup_fail    <= 1'b0;
            run_stall       <= 1'b0;
            chg_cnt[2:0]    <= 3'b0;
            hammer_chk      <= 1'b0;

            // Wait for the calc_updated to assert before doing the calculation
            if(calc_updated) begin
                // Calculate wether the angle we are going to process is small, medium, or large
                if(delta_angle[11:0] < 12'd30)
                    num_steps[3:0] <= SMALL_DELTA; 

                else if(delta_angle[11:0] < 12'd50)
                    num_steps[3:0] <= MED_DELTA; 

                else 
                    num_steps[3:0] <= BIG_DELTA; 

                // Send the first notification to PWM of an update
                pwm_update <= 1'b1;                
            end
        end

        else if(ps == ACCEL) begin
            pwm_ratio[7:0] <= hammer_profile[curr_step[3:0]];

            // Check if the PWM ratio has been absorbed
            if( pwm_done_went_high == 1'b1 ) begin
                // If so, then we can proceed
                profile_delay[7:0] <= profile_delay + 12'h1;

                // If we've waited long enough, then go to the next acceleration step
                if(profile_delay[7:0] == delay_target[7:0]) begin
                    curr_step[3:0] <= curr_step[3:0] + 4'b1;
                    profile_delay[7:0] <= 12'b0;
                end
            end
        end

        // If the hammer profile is enabled, then we use this to try to "unstick" the motor
        else if(ps == HAMMER_FORWARD) begin
            pwm_direction   <= dir_shortest;
            pwm_ratio[7:0]  <= hammer_profile[curr_step[3:0]];

            // Check if the PWM ratio has been absorbed
            if( pwm_done_went_high == 1'b1 ) begin
                // If so, then we can proceed
                profile_delay[7:0] <= profile_delay + 12'h1;

                // If we've waited long enough, then go to the next acceleration step
                if(profile_delay[7:0] == delay_target[7:0]) begin
                    curr_step[3:0] <= curr_step[3:0] + 4'b1;
                    profile_delay[7:0] <= 12'b0;

                    // Check if the angle has changed by > 3
                    if(curr_ang_ff[11:0] > current_angle[11:0]) begin
                        if((curr_ang_ff[11:0] - current_angle[11:0]) >= 12'd3)
                            chg_cnt[2:0]    <= chg_cnt[2:0] + 3'b1;
                    end
                    else begin
                        if((current_angle[11:0] - curr_ang_ff[11:0]) >= 12'd3)
                            chg_cnt[2:0]    <= chg_cnt[2:0] + 3'b1;
                    end
                    curr_ang_ff[11:0]   <= current_angle[11:0];

                    // If we've reached the end of the profile, increment the retry
                    if(curr_step[3:0] == fwd_count[3:0]) begin
                        retry_cnt[1:0] <= retry_cnt[1:0] + 2'b1;
                        curr_step[3:0] <= 4'b0;
                        hammer_chk     <= 1'b1;
                    end
                end                
            end
        end

        // If the we can't get the motor to move forwards, then try to go backwards (before again trying forwards)
        else if(ps == HAMMER_REVERSE) begin
            pwm_direction   <= ~dir_shortest;
            pwm_ratio[7:0]  <= hammer_profile[curr_step[3:0]];

            // Check if the PWM ratio has been absorbed
            if( pwm_done_went_high == 1'b1 ) begin
                // If so, then we can proceed
                profile_delay[7:0] <= profile_delay + 12'h1;

                // If we've waited long enough, then go to the next acceleration step
                if(profile_delay[7:0] == delay_target[7:0]) begin
                    curr_step[3:0] <= curr_step[3:0] + 4'b1;
                    profile_delay[7:0] <= 12'b0;

                    // Check if the angle has changed by > 3
                    if(curr_ang_ff[11:0] > current_angle[11:0]) begin
                        if((curr_ang_ff[11:0] - current_angle[11:0]) < 12'd3)
                            chg_cnt[2:0]    <= chg_cnt[2:0] + 3'b1;
                    end
                    else begin
                        if((current_angle[11:0] - curr_ang_ff[11:0]) > 12'd3)
                            chg_cnt[2:0]    <= chg_cnt[2:0] + 3'b1;
                    end
                    curr_ang_ff[11:0]   <= current_angle[11:0];

                    // If we've reached the end of the profile, reset curr_step
                    if(curr_step[3:0] == rvs_count[3:0]) begin
                        curr_step[3:0] <= 4'b0;
                    end
                end                
            end
        end

        // If hammer wasn't able to start the motor, then disable it to protect
        // both the motor controller and the motor
        else if(ps == HAMMER_FAIL) begin
            startup_fail    <= 1'b1;
            pwm_ratio[7:0]  <= 8'd0;
        end

        else if(ps == CRUISE) begin
            // Continue to run at max speed
            curr_step[3:0]  <= 4'h7;
            pwm_ratio[7:0]  <= linear_profile[curr_step[3:0]];

            // Check for stalls
            if( pwm_done_went_high == 1'b1 ) begin
                // If so, then we can proceed
                profile_delay[7:0] <= profile_delay + 12'h1;

                // If we've waited long enough, then go to the next acceleration step
                if(profile_delay[7:0] == delay_target[7:0]) begin
                    profile_delay[7:0] <= 12'b0;

                    // If the angle hasn't changed by at least 5, then flag a stall
                    if(curr_ang_ff[11:0] > current_angle[11:0]) begin
                        if((curr_ang_ff[11:0] - current_angle[11:0]) < 12'd3)
                            run_stall   <= 1'b1;
                    end
                    else begin
                        if((current_angle[11:0] - curr_ang_ff[11:0]) < 12'd3)
                            run_stall   <= 1'b1;
                    end
                    
                    curr_ang_ff[11:0]   <= current_angle[11:0];
                end
            end
        end

        else if(ps == DECEL ) begin
            pwm_ratio[7:0] <= linear_profile[curr_step[3:0]];

            // Check if the PWM ratio has been absorbed
            if( pwm_done_went_high == 1'b1 ) begin
                // If so, then we can proceed
                profile_delay[7:0] <= profile_delay + 12'h1;

                // If we've waited long enough, then go to the next acceleration step
                if(profile_delay[7:0] == delay_target[7:0]) begin
                    curr_step[3:0] <= curr_step[3:0] - 4'b1;
                    profile_delay[7:0] <= 12'b0;
                end
            end
        end

        else if(ps == SHUTDOWN) begin
            pwm_ratio[7:0] <= 8'b0;
        end

        if((ps == SHUTDOWN) & (ns == IDLE))
            angle_done <= 1'b1;
        else
            angle_done <= 1'b0; 
    end

always @(*) begin
    case(ps)
        IDLE: begin
            if(angle_update)
                ns = CALC;
            else
                ns = IDLE;
        end
        
        CALC: begin
            if(abort_angle)
                ns = IDLE;

            if(calc_updated) begin
                if(delta_angle[11:0] > TARGET_TOLERANCE) begin
                    if(enable_hammer)
                        ns = HAMMER_FORWARD;
                    else
                        ns = ACCEL;
                end
                else
                    ns = IDLE;
            end
            else
                ns = CALC;
        end

        ACCEL: begin
            if(abort_angle)
                ns = DECEL ;
            else if(curr_step[3:0] == 4'hF)
                ns = CRUISE;
            else
                ns = ACCEL;
        end

        HAMMER_FORWARD: begin
            // Check if we've reached the retry threshold, if so
            // then go to HAMMER_FAIL
            if(retry_cnt[1:0] == retry_count[1:0])
                ns = HAMMER_FAIL;
            // if we've hit the target number of changes, then go to HAMMER_PASS
            else if( chg_cnt[2:0] == consec_chg[2:0])
                ns = HAMMER_PASS;
            // if we've reached the fwd_count then go to reverse profile
            else if(hammer_chk) //curr_step[3:0] == fwd_count[3:0])
                ns = HAMMER_REVERSE;
            // Otherwise, we are still in the hammer profile, stay here
            else //if( curr_step < fwd_count[3:0])
                ns = HAMMER_FORWARD;            
        end

        HAMMER_REVERSE: begin
            // if we've reached the fwd_count then go to reverse profile
            if(curr_step[3:0] == rvs_count[3:0])
                ns = HAMMER_FORWARD;
            // Otherwise, we are still in the hammer profile, stay here
            else //if( curr_step < fwd_count[3:0])
                ns = HAMMER_REVERSE;
        end

        HAMMER_PASS: begin
            ns = CRUISE;
        end

        HAMMER_FAIL: begin
            if(angle_update)
                ns = CALC;
            else
                ns = HAMMER_FAIL;
        end

        CRUISE: begin
            if(abort_angle)
                ns = DECEL;
            
            else if(run_stall)
                ns = CALC;

            else if(calc_updated) begin
                // Depending on how large of a delta_angle, we will start decelerating at different points
                if(num_steps[3:0] == SMALL_DELTA)
                    if(delta_angle[11:0] < 12'd5)
                        ns = DECEL ;
                    else
                        ns = CRUISE;
                else if(num_steps[3:0] == MED_DELTA)
                    if(delta_angle[11:0] < 12'd8)
                        ns = DECEL ;
                    else
                        ns = CRUISE;
                else //(num_steps[7:0] == BIG_DELTA)
                    if(delta_angle[11:0] < 12'd10)
                        ns = DECEL ;
                    else
                        ns = CRUISE;
            end
            else
                ns = CRUISE;
        end

        DECEL : begin
            if(delta_angle[11:0] < TARGET_TOLERANCE)
                ns = SHUTDOWN;
            else
                ns = DECEL ;
        end

        SHUTDOWN: begin
            if(pwm_done)
                ns = IDLE;
            else
                ns = SHUTDOWN;
        end

        default: begin 
            ns = IDLE; 
        end 
    endcase
end

calculate_delta calc (
    .reset_n        (reset_n),      
    .clock          (clock),        
    .enable_calc    (enable_calc),  
    .target_angle   (target_angle[11:0]), 
    .current_angle  (current_angle[11:0]),
    .dir_shortest   (dir_shortest), 
    .delta_angle    (delta_angle[11:0]),  
    .calc_updated   (calc_updated)
);

endmodule


