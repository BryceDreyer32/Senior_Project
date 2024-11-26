// Copyright 2024
// Bryce's Senior Project
// Description: This is the RTL for "Angle to PWM". This will include calculating the PWM control values based on current and target angles
//              and includes the acceleration and deceleration profiles. 

module angle_to_pwm(
    input               reset_n,        // Active low reset
    input               clock,          // The main clock
    input   [11:0]      target_angle,   // The angle the wheel needs to move to in degrees. This number is multiplied by 2 internally
    input   [11:0]      current_angle,  // The angle read from the motor encoder
    input               pwm_done,       // Indicator from PWM that the pwm_ratio has been applied
    input               angle_update,   // Request to update the angle
    output reg          angle_done,     // Indicator that the angle has been applied 
    output reg          pwm_enable,     // PWM enable
    output reg          pwm_update,     // Request an update to the PWM ratio
    output reg [7:0]    pwm_ratio,      // The high-time of the PWM signal out of 255.
    output              pwm_direction   // The direction of the motor
);

// States
localparam IDLE   = 2'd0;
localparam ACCEL  = 2'd1;
localparam CRUISE = 2'd2;
localparam DECCEL = 2'd3; 

localparam SMALL_DELTA  = 4'd8;
localparam MED_DELTA    = 4'd11;
localparam BIG_DELTA    = 4'd15;

localparam PROFILE_DELAY_TARGET = 12'd3;
localparam TARGET_TOLERANCE     = 13'd2; 

reg [1:0]   ps, ns;
reg [7:0]   profile         [15:0];
reg [12:0]  delta_angle;
reg [3:0]   num_steps;
reg [3:0]   curr_step;
reg [11:0]  profile_delay;


// Initialize the profile
always @(negedge reset_n) begin
    profile[0][7:0]  = 8'd6;
    profile[1][7:0]  = 8'd18;
    profile[2][7:0]  = 8'd29;
    profile[3][7:0]  = 8'd39;
    profile[4][7:0]  = 8'd49;
    profile[5][7:0]  = 8'd59;
    profile[6][7:0]  = 8'd68;
    profile[7][7:0]  = 8'd76;
    profile[8][7:0]  = 8'd84;
    profile[9][7:0]  = 8'd91;
    profile[10][7:0] = 8'd98;
    profile[11][7:0] = 8'd104;
    profile[12][7:0] = 8'd110;
    profile[13][7:0] = 8'd115;
    profile[14][7:0] = 8'd119;
    profile[15][7:0] = 8'd123;
end

always @(posedge clock or negedge reset_n)
    if(~reset_n) begin
        ps <= IDLE;
        delta_angle[12:0]   <= 13'b0;
        curr_step[3:0]      <= 4'b0;
        pwm_ratio[7:0]      <= 8'd128;
        pwm_enable          <= 1'b1;
        pwm_update          <= 1'b0;
        profile_delay[11:0] <= 12'b0;
        angle_done          <= 1'b0;
        num_steps[3:0]      <= MED_DELTA;
    end
    else begin
        ps <= ns;
        
        // If target_angle > current_angle then set delta_angle[12] to 0 otherwise set to 1
        if(target_angle[11:0] >= current_angle[11:0]) 
            delta_angle[12:0] <= {1'b0, target_angle[11:0] - current_angle[11:0]};
        else
            delta_angle[12:0] <= {1'b1, current_angle[11:0] - target_angle[11:0]};

        if(ps == IDLE) begin
            // If we are in IDLE force the ratio to 128
            curr_step[3:0] <= 4'b0;
            pwm_ratio[7:0] <= 8'd128;
            if(pwm_done == 1'b0)
                pwm_update <= 1'b1;
            else 
                pwm_update <= 1'b0;

            //Calculate wether the angle we are going to process is small, medium, or large
            if(delta_angle[12:0] < 13'd10)
                num_steps[3:0] = SMALL_DELTA; 

            else if(delta_angle[12:0] < 13'd30)
                num_steps[3:0] = MED_DELTA; 

            else 
                num_steps[3:0] = BIG_DELTA; 
        end

        if(ps == ACCEL) begin
            // If the delta_angle is negative then subtract the acceleration value from 128
            if(delta_angle[12] == 1'b1)
                pwm_ratio[7:0] <= 8'd128 - profile[curr_step[3:0]];
            
            // Otherwise we add the acceleration value to 128
            else
                pwm_ratio[7:0] <= 8'd128 + profile[curr_step[3:0]];

            if(pwm_done == 1'b0)
                pwm_update <= 1'b1;

            else begin
                pwm_update <= 1'b0;
                profile_delay[11:0] <= profile_delay + 12'h1;
            end

            if(profile_delay[11:0] == PROFILE_DELAY_TARGET) begin
                curr_step[3:0] <= curr_step[3:0] + 4'b1;
                profile_delay[11:0] <= 12'b0;
            end
        end

        else if(ps == DECCEL) begin
            // If the delta_angle is negative then subtract the decceleration value from 128
            if(delta_angle[12] == 1'b1)
                pwm_ratio[7:0] <= 8'd128 - profile[curr_step[3:0]];
            
            // Otherwise we add the acceleration value to 128
            else
                pwm_ratio[7:0] <= 8'd128 + profile[curr_step[3:0]];

            if(~pwm_done)
                pwm_update <= 1'b1;

            else begin
                pwm_update <= 1'b0;
                profile_delay[11:0] <= profile_delay + 12'h1;
            end

            // Decelerate until we hit the minimum value (avoid roll over)
            if(profile_delay[11:0] == PROFILE_DELAY_TARGET) begin
                if(curr_step[3:0] >= 4'b1)
                    curr_step[3:0] <= curr_step[3:0] - 4'b1;
                profile_delay[11:0] <= 12'b0;
            end
        end

        if((ps == DECCEL) & (ns == IDLE))
            angle_done <= 1'b1;
        else
            angle_done <= 1'b0; 
    end

always @(*) begin
    case(ps)
        IDLE: begin
            if((delta_angle[12:0] > TARGET_TOLERANCE) & angle_update)
                ns = ACCEL;
            else
                ns = IDLE;
        end
        
        ACCEL: begin
            if(curr_step[3:0] == num_steps[3:0])
                ns = CRUISE;
            else
                ns = ACCEL;
        end

        CRUISE: begin
            // Depending on how large of a delta_angle, we will start decelerating at different points
            if(num_steps[3:0] == SMALL_DELTA)
                if(delta_angle[12:0] < 13'd4)
                    ns = DECCEL;
                else
                    ns = CRUISE;
            else if(num_steps[3:0] == MED_DELTA)
                if(delta_angle[12:0] < 13'd6)
                    ns = DECCEL;
                else
                    ns = CRUISE;
            else //(num_steps[3:0] == BIG_DELTA)
                if(delta_angle[12:0] < 13'd8)
                    ns = DECCEL;
                else
                    ns = CRUISE;
        end

        DECCEL: begin
            if(delta_angle[12:0] < TARGET_TOLERANCE)
                ns = IDLE;
            else
                ns = DECCEL;
        end

        default: begin 
            ns = IDLE; 
        end 
    endcase
end
endmodule


