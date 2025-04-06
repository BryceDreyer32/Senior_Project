// Copyright 2024
// Bryce's Senior Project
// Description: This is the RTL for servo control
module servo_ctrl 
(
    input       reset_n,            // Active low reset
    input       clock,              // The main clock
    input       pwm_enable,         // Enables the PWM output
    input [7:0] start_pwm_ratio,    // The initial pwm ratio
    input [7:0] target_pwm_ratio,   // The target pwm ratio
    output      pwm_signal          // The output PWM wave
);
    
// States
localparam IDLE             = 3'd0;
localparam GO_TO_START      = 3'd1;
localparam CALCULATE_DELTA  = 3'd2;
localparam ACCELERATE       = 3'd3;
localparam CRUISE           = 3'd4;
localparam DECELERATE       = 3'd5;
localparam HOLD             = 3'd6;


wire    [7:0]   pwm_ratio;           // The high-time of the PWM signal out of 255.
wire    [7:0]   linear_profile       [7:0];
reg     [7:0]   adjusted_pwm_ratio;
wire            pwm_done;
reg     [2:0]   curr_step;
reg     [7:0]   curr_pwm_ratio;
reg     [7:0]   profile_delay;
reg     [2:0]   state;
reg             accel_flag;
reg             direction;
reg             pwm_done_ff, pwm_done_went_high;

// Initialize the acceleration and deceleration profiles
assign linear_profile[0][7:0]  = 8'h70;
assign linear_profile[1][7:0]  = 8'h60;
assign linear_profile[2][7:0]  = 8'h50;
assign linear_profile[3][7:0]  = 8'h40;
assign linear_profile[4][7:0]  = 8'h30;
assign linear_profile[5][7:0]  = 8'h28;
assign linear_profile[6][7:0]  = 8'h20;
assign linear_profile[7][7:0]  = 8'h18;


always @( posedge clock or negedge reset_n ) begin
    if( ~reset_n ) begin
        curr_step           <= 3'h0;
        curr_pwm_ratio      <= 8'h0;
        profile_delay       <= 8'h0;
        accel_flag          <= 1'b0;
        direction           <= 1'b0;
        state               <= IDLE;
        pwm_done_ff         <= 1'b0;
        pwm_done_went_high  <= 1'b0;        
    end

    else begin
        pwm_done_ff         <= pwm_done;
        pwm_done_went_high  <= ~pwm_done_ff & pwm_done;

        case(state)
            IDLE: begin
                if(pwm_enable) begin
                    curr_pwm_ratio      <= start_pwm_ratio;
                    state               <= GO_TO_START;
                    // If the tartget pwm ratio = 0, then set the pwm ratio to start pwm ratio
                    if(target_pwm_ratio == 8'h0)
                        adjusted_pwm_ratio  <= start_pwm_ratio;
                    else
                        adjusted_pwm_ratio  <= target_pwm_ratio;
                end
            end

            GO_TO_START: begin
                if(profile_delay[7:0] == 8'h20) begin
                    state           <= CALCULATE_DELTA;
                    profile_delay   <= 8'h0;
                end
                else begin
                    if( pwm_done_went_high == 1'b1 ) 
                        profile_delay   <= profile_delay + 8'h1;
                end
            end

            CALCULATE_DELTA: begin
                if(adjusted_pwm_ratio > curr_pwm_ratio) begin
                    direction       <= 1'b1;

                    if(adjusted_pwm_ratio - curr_pwm_ratio > 8'd16) begin
                        state       <= ACCELERATE;
                        accel_flag  <= 1'b1;
                    end
                    else begin
                        state       <= CRUISE;
                        accel_flag  <= 1'b0;
                    end
                end
                else if(adjusted_pwm_ratio <= curr_pwm_ratio) begin
                    direction       <= 1'b0;

                    if(curr_pwm_ratio - adjusted_pwm_ratio > 8'd16) begin
                        state       <= ACCELERATE;
                        accel_flag  <= 1'b1;
                    end
                    else begin
                        state       <= CRUISE;
                        accel_flag  <= 1'b0;
                    end
                end
            end

            ACCELERATE: begin
                if(curr_step < 3'h7) begin
                    if( pwm_done_went_high == 1'b1 ) begin
                        if(profile_delay[7:0] == linear_profile[curr_step]) begin
                            curr_step       <= curr_step + 3'h1;
                            profile_delay   <= 8'h0;

                            if(direction) begin
                                curr_pwm_ratio <= curr_pwm_ratio + 8'h1;
                            end
                            else begin
                                curr_pwm_ratio <= curr_pwm_ratio - 8'h1;
                            end
                        end
                        else begin
                            profile_delay   <= profile_delay + 8'h1;
                        end
                    end
                end
                else begin
                    state           <= CRUISE;
                    profile_delay   <= 8'h0;
                end
            end

            CRUISE: begin
                if(accel_flag) begin
                    if( pwm_done_went_high == 1'b1 ) begin
                        if(profile_delay[7:0] == linear_profile[7]) begin
                            if(direction == 1'b0) begin
                                if(curr_pwm_ratio - adjusted_pwm_ratio < 8) begin
                                    state           <= DECELERATE;
                                    profile_delay   <= 8'h0;
                                    curr_step       <= 3'h6;
                                end 
                                else begin
                                    curr_pwm_ratio  <= curr_pwm_ratio - 8'h1;
                                    profile_delay   <= 8'h0;
                                end
                            end
                            else begin
                                if(adjusted_pwm_ratio - curr_pwm_ratio < 8) begin
                                    state           <= DECELERATE;
                                    profile_delay   <= 8'h0;
                                    curr_step       <= 3'h6;
                                end 
                                else begin
                                    curr_pwm_ratio  <= curr_pwm_ratio + 8'h1;
                                    profile_delay   <= 8'h0;
                                end
                            end
                        end
                        else begin
                            profile_delay   <= profile_delay + 8'h1;
                        end
                    end
                end

                else begin
                    if(curr_pwm_ratio == adjusted_pwm_ratio) begin
                        state           <= HOLD;
                    end
                    else begin
                        if( pwm_done_went_high == 1'b1 ) begin
                            if(profile_delay[7:0] == linear_profile[7]) begin
                                if(direction == 1'b0)
                                    curr_pwm_ratio  <= curr_pwm_ratio - 8'h1;
                                else 
                                    curr_pwm_ratio  <= curr_pwm_ratio + 8'h1;
                                profile_delay   <= 8'h0;
                            end
                            else begin
                                profile_delay       <= profile_delay + 8'h1;
                            end
                        end
                    end
                end
            end

            DECELERATE: begin
                if(curr_step > 3'h0) begin
                    if( pwm_done_went_high == 1'b1 ) begin
                        if(profile_delay[7:0] == linear_profile[curr_step]) begin
                            curr_step       <= curr_step - 3'h1;
                            profile_delay   <= 8'h0;
                            
                            if(direction) begin
                                curr_pwm_ratio <= curr_pwm_ratio + 8'h1;
                            end
                            else begin
                                curr_pwm_ratio <= curr_pwm_ratio - 8'h1;
                            end
                        end
                        else begin
                            profile_delay   <= profile_delay + 8'h1;
                        end
                    end
                end
                else begin
                    state           <= HOLD;
                    profile_delay   <= 8'h0;
                end
            end

            HOLD: begin
                profile_delay   <= 8'h0;
                if(pwm_enable == 1'b0) begin
                    state <= IDLE;
                end
                else begin
                    if(target_pwm_ratio == 8'h0)
                        adjusted_pwm_ratio  <= start_pwm_ratio;
                    else
                        adjusted_pwm_ratio  <= target_pwm_ratio;  
                                          
                    if(curr_pwm_ratio != adjusted_pwm_ratio) begin
                        state <= CALCULATE_DELTA;
                    end
                end
            end

            default: begin
                state <= IDLE;
            end
        endcase
    end
end

pwm servo_pwm(
    .reset_n                (reset_n),
    .clock                  (clock),
    .pwm_enable             (pwm_enable),
    .pwm_ratio              (curr_pwm_ratio[7:0]),
    .pwm_update             (1'b1),
    .pwm_done               (pwm_done),
    .pwm_signal             (pwm_signal)
);

endmodule