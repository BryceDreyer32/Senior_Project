// Copyright 2025
// Bryce's Senior Project
// Description: This is the Proportional-Integral-Derivative (PID) controller for the motor

module pid(
    input               reset_n,            // Active low reset
    input               clock,              // The main clock
    input       [11:0]  target_angle,       // The angle the wheel needs to move on the 4096 points/rotation scale
    input       [11:0]  current_angle,      // The angle read from the motor encoder
    input               pwm_enable,         // PWM enable
    input               pwm_done,           // Indicator from PWM that the pwm_ratio has been applied
    input               i2c_rd_done,        // Read done from I2C 
    input               angle_update,       // Request to update the angle
    input               abort_angle,        // Aborts rotating to angle

    input               enable_stall_chk,   // Enable the stall check
    output reg          stalled,            // Error: Motor stalled, unable to startup
    input       [7:0]   kp,                 // Proportional Constant: fixed point 4.4
    input       [3:0]   ki,                 // Integral Constant: fixed point 0.4
    input       [3:0]   kd,                 // Derivative Constant: fixed point 0.4

    output      [15:0]  debug_signals,  
    output reg          angle_done,         // Indicator that the angle has been applied 
    output reg          pwm_update,         // Request an update to the PWM ratio
    output reg  [7:0]   pwm_ratio,          // The high-time of the PWM signal out of 255.
    output              pwm_direction       // The direction of the motor
);

localparam          IDLE    = 2'b00;
localparam          ACCEL   = 2'b01;
localparam          CRUISE  = 2'b10;
localparam          DECEL   = 2'b11; 

reg     [1:0]       state;
reg     [5:0]       curr_step;
wire    [31:0]      proportional_error;             // The proportional instantenous error
wire    [31:0]      integral_error;                 // The cumulative error
wire    [31:0]      derivative_error;               // The derivative error
reg     [15:0]      elapsed_time;                   // The elapsed time since the last update
wire    [19:0]      delta_12p8;                     // Delta in Fixed point 12.4 format
reg     [19:0]      last_delta_12p8;                // The last error from the PID controller
wire    [11:0]      delta_angle;                    // The angle difference between the target and current angle
wire                calc_updated;                   // Delta angle has been updated
reg                 rd_done, i2c_rd_done_ff;        // Read done
wire    [15:0]      ratio_int;

// To convert from regular binary to FP 12.8 format, we shift to the left 8 times
// because the lower 8 bits are storing the decimal portion
assign delta_12p8   = delta_angle << 8;
assign debug_signals[15:0] = {7'b0, curr_step[5:0], pwm_direction, state[1:0]};

always @(negedge reset_n or posedge clock) begin
    if(~reset_n) begin
        angle_done          <= 1'b0; 
        pwm_update          <= 1'b0; 
        stalled             <= 1'b0; 
        elapsed_time        <= 16'b1;
        last_delta_12p8     <= 20'b0;
        rd_done             <= 1'b0;
        i2c_rd_done_ff      <= 1'b0;
        state               <= IDLE;
        curr_step           <= 6'b0;
        pwm_ratio           <= 8'b0;
    end
    else begin
        i2c_rd_done_ff  <= i2c_rd_done;
        rd_done         <= !i2c_rd_done_ff & i2c_rd_done; 
        
        case (state)
            IDLE: begin
                if(angle_update) begin
                    state       <= ACCEL;
                    angle_done  <= 1'b0;
                end
                pwm_update  <= 1'b0; 
                curr_step   <= 6'b0;
                pwm_ratio   <= 8'b0;
            end

            ACCEL: begin
                if(~pwm_enable)
                    state <= IDLE;

                pwm_update  <= 1'b1; 
                if(rd_done) begin
                    if(curr_step[5:3] <= 3'b111) begin
                        case(curr_step)
                            3'b000, 3'b001, 3'b010: pwm_ratio   <= ratio_int >> 8;
                            3'b011, 3'b100: pwm_ratio           <= ratio_int >> 7;
                            3'b101, 3'b110: pwm_ratio           <= ratio_int >> 6;
                            default:        pwm_ratio           <= ratio_int >> 5;   
                        endcase
                    end

                    if(curr_step[5:0] < 6'b111111) 
                        curr_step <= curr_step + 6'b1;    
                    else
                        state <= CRUISE;
                end
            end 

            CRUISE: begin
                if(~pwm_enable)
                    state <= IDLE;

                if(delta_angle < 12'd50)
                    state <= DECEL;
                
                pwm_ratio <= ratio_int >> 5;
            end 

            DECEL: begin
                if(~pwm_enable)
                    state <= IDLE;

                if(delta_angle < 12'd10) begin
                    state       <= IDLE;
                    angle_done  <= 1'b1;
                    pwm_ratio   <= ratio_int >> 8;
                end

                else 
                    pwm_ratio <= ratio_int >> 5;
            end 

            default: 
                state <= IDLE;

        endcase


        if (rd_done) begin
            elapsed_time            <= elapsed_time + 16'b1;
            last_delta_12p8         <= delta_12p8;
        end 
    end
end

assign proportional_error       = (kp << 4) * delta_12p8;
assign integral_error           = '0;//ki * ((delta_12p8 >> 12) & 12'hFFF) * ((elapsed_time >> 12) & 12'hFFF);
assign derivative_error         = '0;//kd * (delta_12p8 - last_delta_12p8) / elapsed_time;
assign ratio_int                = ((proportional_error + integral_error + derivative_error) >> 16);// & 16'hFFF;


calculate_delta calc (
    .reset_n        (reset_n),      
    .clock          (clock),        
    .enable_calc    (rd_done),  
    .target_angle   (target_angle[11:0]), 
    .current_angle  (current_angle[11:0]),
    .dir_shortest   (pwm_direction), 
    .delta_angle    (delta_angle[11:0]),  
    .calc_updated   (calc_updated)
);
endmodule
