// Copyright 2024
// Bryce's Senior Project
// Description: This is the RTL for "Angle to PWM". This will include calculating the PWM control values based on current and target angles
//              and includes the acceleration and deceleration profiles. 

module angle_to_pwm(
    input               reset_n,        // Active low reset
    input               clock,          // The main clock
    input   [11:0]      target_angle,   // The angle the wheel needs to move on the 4096 points/rotation scale
    input   [11:0]      current_angle,  // The angle read from the motor encoder
    input               pwm_done,       // Indicator from PWM that the pwm_ratio has been applied
    input               angle_update,   // Request to update the angle
    input               abort_angle,    // Aborts rotating to angle
    output  [7:0]       debug_signals,
    output reg          angle_done,     // Indicator that the angle has been applied 
    output reg          pwm_enable,     // PWM enable
    output reg          pwm_update,     // Request an update to the PWM ratio
    output reg [7:0]    pwm_ratio,      // The high-time of the PWM signal out of 255.
    output reg          pwm_direction   // The direction of the motor
);

// States
localparam IDLE     = 3'd0;
localparam CALC     = 3'd1;
localparam ACCEL    = 3'd2;
localparam CRUISE   = 3'd3;
localparam DECCEL   = 3'd4; 
localparam SHUTDOWN = 3'd5; 

localparam SMALL_DELTA  = 4'd8;
localparam MED_DELTA    = 4'd10;
localparam BIG_DELTA    = 4'd14;

localparam PROFILE_DELAY_TARGET = 12'd2;
localparam TARGET_TOLERANCE     = 12'd5; 

reg   [2:0] ps, ns;
wire  [7:0] profile         [15:0];
wire [11:0] delta_angle;
reg   [3:0] num_steps;
reg   [3:0] curr_step;
reg  [11:0] profile_delay;
wire        calc_updated;
reg         enable_calc;

assign debug_signals = {1'b0, angle_update, abort_angle, pwm_done, pwm_update, ps[2:0]};

// Initialize the profile
assign profile[0][7:0]  = 8'd11;
assign profile[1][7:0]  = 8'd29;
assign profile[2][7:0]  = 8'd48;
assign profile[3][7:0]  = 8'd65;
assign profile[4][7:0]  = 8'd82;
assign profile[5][7:0]  = 8'd99;
assign profile[6][7:0]  = 8'd113;
assign profile[7][7:0]  = 8'd128;
assign profile[8][7:0]  = 8'd141;
assign profile[9][7:0]  = 8'd153;
assign profile[10][7:0] = 8'd164;
assign profile[11][7:0] = 8'd174;
assign profile[12][7:0] = 8'd185;
assign profile[13][7:0] = 8'd193;
assign profile[14][7:0] = 8'd200;
assign profile[15][7:0] = 8'd206;


always @(posedge clock or negedge reset_n)
    if(~reset_n) begin
        ps <= IDLE;
        curr_step[3:0]      <= 4'b0;
        pwm_ratio[7:0]      <= 8'd0;
        pwm_enable          <= 1'b1;
        pwm_update          <= 1'b0;
        profile_delay[11:0] <= 12'b0;
        angle_done          <= 1'b0;
        enable_calc         <= 1'b0;
        num_steps[3:0]      <= MED_DELTA;
    end
    else begin
        ps                  <= ns;
        pwm_enable          <= 1'b1;
        
        if(ps == IDLE) begin
            // If we are in IDLE force the ratio to 0
            curr_step[3:0]  <= 4'b0;
            pwm_ratio[7:0]  <= 8'd0;
            pwm_update      <= 1'b0;
            enable_calc     <= 1'b0;
        end

        else if( ps == CALC) begin
            enable_calc     <= 1'b1;

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
            pwm_ratio[7:0] <= profile[curr_step[3:0]];

            // Check if the PWM ratio has been absorbed
            if( pwm_done == 1'b1 ) begin
                // If so, then we can proceed
                profile_delay[11:0] <= profile_delay + 12'h1;

                // If we've waited long enough, then go to the next acceleration step
                if(profile_delay[11:0] == PROFILE_DELAY_TARGET) begin
                    curr_step[3:0] <= curr_step[3:0] + 4'b1;
                    profile_delay[11:0] <= 12'b0;
                end
            end
        end

        else if(ps == CRUISE) begin
            // Continue to run at max speed
            curr_step[3:0]  <= 4'hF;
            pwm_ratio[7:0]  <= profile[curr_step[3:0]];
        end

        else if(ps == DECCEL) begin
            pwm_ratio[7:0] <= profile[curr_step[3:0]];

            // Check if the PWM ratio has been absorbed
            if( pwm_done == 1'b1 ) begin
                // If so, then we can proceed
                profile_delay[11:0] <= profile_delay + 12'h1;

                // If we've waited long enough, then go to the next acceleration step
                if(profile_delay[11:0] == PROFILE_DELAY_TARGET) begin
                    curr_step[3:0] <= curr_step[3:0] - 4'b1;
                    profile_delay[11:0] <= 12'b0;
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
                if(delta_angle[11:0] > TARGET_TOLERANCE)
                    ns = ACCEL;
                else
                    ns = IDLE;
            end
            else
                ns = CALC;
        end

        ACCEL: begin
            if(abort_angle)
                ns = DECCEL;
            else if(curr_step[3:0] == num_steps[3:0])
                ns = CRUISE;
            else
                ns = ACCEL;
        end

        CRUISE: begin
            if(abort_angle)
                ns = DECCEL;

            if(calc_updated) begin
                // Depending on how large of a delta_angle, we will start decelerating at different points
                if(num_steps[3:0] == SMALL_DELTA)
                    if(delta_angle[11:0] < 12'd5)
                        ns = DECCEL;
                    else
                        ns = CRUISE;
                else if(num_steps[3:0] == MED_DELTA)
                    if(delta_angle[11:0] < 12'd8)
                        ns = DECCEL;
                    else
                        ns = CRUISE;
                else //(num_steps[7:0] == BIG_DELTA)
                    if(delta_angle[11:0] < 12'd10)
                        ns = DECCEL;
                    else
                        ns = CRUISE;
            end
            else
                ns = CRUISE;
        end

        DECCEL: begin
            if(delta_angle[11:0] < TARGET_TOLERANCE)
                ns = SHUTDOWN;
            else
                ns = DECCEL;
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


