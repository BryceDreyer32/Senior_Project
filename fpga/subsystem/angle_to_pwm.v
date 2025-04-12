// Copyright 2024
// Bryce's Senior Project
// Description: This is the RTL for "Angle to PWM". This will include calculating the PWM control values based on current and target angles
//              and includes the acceleration and deceleration profiles. 

module angle_to_pwm(
    input               reset_n,            // Active low reset
    input               clock,              // The main clock
    input       [11:0]  target_angle,       // The angle the wheel needs to move on the 4096 points/rotation scale
    input       [11:0]  current_angle,      // The angle read from the motor encoder
    input               pwm_enable,         // PWM enable
    input               pwm_done,           // Indicator from PWM that the pwm_ratio has been applied
    input               angle_update,       // Request to update the angle
    input               abort_angle,        // Aborts rotating to angle

    input               enable_stall_chk,   // Enable the stall check
    input       [7:0]   delay_target,       // Number of times to remain on each profile step
    input       [7:0]   profile_offset,     // An offset that is added to each of the profile steps
    input       [7:0]   cruise_power,       // The amount of power to apply during the cruise phase
    output reg          startup_fail,       // Error: Motor stalled, unable to startup
    input       [127:0] pwm_profile,        // 16 * 8 bit pwm profile 


    output      [15:0]  debug_signals,  
    output reg          angle_done,         // Indicator that the angle has been applied 
    output reg          pwm_update,         // Request an update to the PWM ratio
    output reg  [7:0]   pwm_ratio,          // The high-time of the PWM signal out of 255.
    output reg          pwm_direction       // The direction of the motor
);

// States
localparam IDLE         = 3'd0;
localparam CALC         = 3'd1;
localparam ACCEL        = 3'd2;
localparam CRUISE       = 3'd3;
localparam DECEL        = 3'd4; 
localparam SHUTDOWN     = 3'd5; 

localparam SMALL_DELTA  = 4'd8;
localparam MED_DELTA    = 4'd10;
localparam BIG_DELTA    = 4'd14;

localparam TARGET_TOLERANCE     = 12'd32; 

reg   [2:0] state;
wire  [7:0] linear_profile               [15:0];
wire [11:0] delta_angle;
reg  [11:0] curr_ang_ff;
reg   [3:0] num_steps;
reg   [3:0] curr_step;
reg  [23:0] profile_delay;
wire [23:0] profile_delay_target;
wire        calc_updated;
wire        dir_shortest;
reg         enable_calc;
reg         pwm_done_ff, pwm_done_went_high;
reg         run_stall;
reg         wentCALC, wentACCEL, wentCRUISE;

assign debug_signals = {startup_fail, run_stall, pwm_enable, 1'b0,/*retry_cnt[1:0],*/ pwm_direction, angle_update, abort_angle, pwm_done,
                        wentCRUISE, wentACCEL, wentCALC, pwm_update, enable_stall_chk, state[2:0]};

assign profile_delay_target[23:0] = delay_target[7:4] << delay_target[3:0];

assign linear_profile[0][7:0]  = pwm_profile[1*8-1:0*8] + profile_offset[7:0];
assign linear_profile[1][7:0]  = pwm_profile[2*8-1:1*8] + profile_offset[7:0];
assign linear_profile[2][7:0]  = pwm_profile[3*8-1:2*8] + profile_offset[7:0];
assign linear_profile[3][7:0]  = pwm_profile[4*8-1:3*8] + profile_offset[7:0];
assign linear_profile[4][7:0]  = pwm_profile[5*8-1:4*8] + profile_offset[7:0];
assign linear_profile[5][7:0]  = pwm_profile[6*8-1:5*8] + profile_offset[7:0];
assign linear_profile[6][7:0]  = pwm_profile[7*8-1:6*8] + profile_offset[7:0];
assign linear_profile[7][7:0]  = pwm_profile[8*8-1:7*8] + profile_offset[7:0];
assign linear_profile[8][7:0]  = pwm_profile[9*8-1:8*8] + profile_offset[7:0];
assign linear_profile[9][7:0]  = pwm_profile[10*8-1:9*8] + profile_offset[7:0];
assign linear_profile[10][7:0] = pwm_profile[11*8-1:10*8] + profile_offset[7:0];
assign linear_profile[11][7:0] = pwm_profile[12*8-1:11*8] + profile_offset[7:0];
assign linear_profile[12][7:0] = pwm_profile[13*8-1:12*8] + profile_offset[7:0];
assign linear_profile[13][7:0] = pwm_profile[14*8-1:13*8] + profile_offset[7:0];
assign linear_profile[14][7:0] = pwm_profile[15*8-1:14*8] + profile_offset[7:0];
assign linear_profile[15][7:0] = pwm_profile[16*8-1:15*8] + profile_offset[7:0];


always @(negedge reset_n or posedge clock) begin
    if(~reset_n) begin
        state               <= IDLE;
        curr_step[3:0]      <= 4'b0;
        curr_ang_ff[11:0]   <= 12'b0;
        pwm_ratio[7:0]      <= 8'd0;
        pwm_update          <= 1'b0;
        pwm_done_ff         <= 1'b0;
        pwm_done_went_high  <= 1'b0;
        profile_delay[23:0] <= 24'b0;
        angle_done          <= 1'b0;
        enable_calc         <= 1'b0;
        num_steps[3:0]      <= MED_DELTA;
        pwm_direction       <= 1'b0;
        startup_fail        <= 1'b0;
        run_stall           <= 1'b0;
        wentCALC            <= 1'b0;
        wentACCEL           <= 1'b0; 
        wentCRUISE          <= 1'b0;
    end
    else begin       
        if(~pwm_enable /*| abort_angle*/)
            state <= IDLE;

        // The pwm_done_went_high ensures that the PWM has had reset this signal
        // and it is now going high. This avoids a race condition where the ratio
        // is updated, and the pwm_done is immediately read: it may not have had 
        // time to reset, and therefore gives a false impression the new ratio
        // has been absorbed when in reality it hasn't.
        pwm_done_ff         <= pwm_done;
        pwm_done_went_high  <= ~pwm_done_ff & pwm_done;

        case(state)
            IDLE: begin
                // If we are in IDLE force the ratio to 0
                angle_done      <= 1'b0;
                curr_step[3:0]  <= 4'b0;
                pwm_ratio[7:0]  <= 8'd0;
                pwm_direction   <= dir_shortest;
                pwm_update      <= 1'b0;
                enable_calc     <= 1'b0;
                
                // State transitions
                if(angle_update)
                    state <= CALC;
            end

            CALC: begin
                curr_step[3:0]  <= 4'b0;
                pwm_ratio[7:0]  <= 8'd0;
                pwm_direction   <= dir_shortest;
                enable_calc     <= 1'b1;
                startup_fail    <= 1'b0;
                run_stall       <= 1'b0;
                
                wentCALC        <= 1'b1;

                // Wait for the calc_updated to assert before doing the calculation
                if(calc_updated) begin
                    if(delta_angle[11:0] > TARGET_TOLERANCE)
                        state   <=  ACCEL;
                    
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

            ACCEL: begin
                wentACCEL   <= 1'b1;

                pwm_ratio[7:0] <= linear_profile[curr_step[3:0]];

                // Check if the PWM ratio has been absorbed
                if( pwm_done_went_high == 1'b1 ) begin
                    // If so, then we can proceed
                    profile_delay[23:0] <= profile_delay[23:0] + 24'h1;

                    // If we've waited long enough, then go to the next acceleration step
                    if(profile_delay[23:0] == profile_delay_target[23:0]) begin                    
                        if(curr_step[3:0] == num_steps[3:0])
                            state <= CRUISE;
                        else
                            curr_step[3:0]  <= curr_step[3:0] + 4'b1;

                        profile_delay[23:0] <= 24'b0;
                        curr_ang_ff[11:0]   <= current_angle[11:0];
                    end
                end
            end

            CRUISE: begin
                wentCRUISE <= 1'b1;
                // Set the current step to match the delta angle size (and hence num_steps)
                curr_step[3:0] <= num_steps[3:0];

                // Continue to run at max speed
                pwm_ratio[7:0]  <= cruise_power[7:0];

                // Check for stalls
                if( pwm_done_went_high == 1'b1 ) begin
                    // If so, then we can proceed
                    profile_delay[23:0] <= profile_delay[23:0] + 24'h1;

                    // Below check is looking to ensure that movement is occuring, if not a stall may be flagged
                    if(profile_delay[23:0] == profile_delay_target[23:0]) begin
                        profile_delay[23:0] <= 24'b0;

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

                // If stall check is enabled, and a stall was detected, then go back to IDLE state
                // to protect the motor and the controller
                if(run_stall & enable_stall_chk)
                    state <=  IDLE;

                if(delta_angle[11:0] < 12'd50) begin
                    state <=  DECEL;
                end
/*                else if(calc_updated) begin
                    // Depending on how large of a delta_angle, we will start decelerating at different points
                    if(num_steps[3:0] == SMALL_DELTA)
                        if(delta_angle[11:0] < 12'd50)
                            state <=  DECEL;
                    else if(num_steps[3:0] == MED_DELTA)
                        if(delta_angle[11:0] < 12'd80)
                            state <=  DECEL;
                    else //(num_steps[7:0] == BIG_DELTA)
                        if(delta_angle[11:0] < 12'd100)
                            state <=  DECEL;
                end
*/            end

            DECEL: begin
                pwm_ratio[7:0] <= linear_profile[curr_step[3:0]];

                // Check if the PWM ratio has been absorbed
                if( pwm_done_went_high == 1'b1 ) begin
                    // If so, then we can proceed
                    profile_delay[23:0] <= profile_delay[23:0] + 24'h1;

                    // If we've waited long enough, then go to the next acceleration step
                    // Note that the below is going to stop the motor "wherever" - hopefully 
                    // close to the target, but not necessarily exactly at it
                    if(profile_delay[23:0] == profile_delay_target[23:0]) begin
                        curr_step[3:0] <= curr_step[3:0] - 4'b1;
                        if(curr_step[3:0] == 4'b0)
                            state <= SHUTDOWN;
                        profile_delay[23:0] <= 24'b0;
                    end
                end

                // The below is a short-circuit path to stop the motor - if we're within tolerance/2
                // then we are just going to cut power entirely
                if(delta_angle[11:0] < (TARGET_TOLERANCE<<1))
                    state <= SHUTDOWN;
            end

            SHUTDOWN: begin
                pwm_ratio[7:0]  <= 8'b0;
                angle_done      <= 1'b1;

                if(pwm_done_went_high == 1'b1)
                    state <= IDLE;
            end
            
            default: begin
                state <= IDLE;
            end
        endcase
    end
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


