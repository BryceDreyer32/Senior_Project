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
    input       [63:0]  profile,            // Trapezoidal profile in fixed point 4.4 notation

    input               enable_stall_chk,   // Enable the stall check
    output reg          stalled,            // Error: Motor stalled, unable to startup
    input       [7:0]   kp,                 // Proportional Constant: fixed point 4.4
    input       [3:0]   ki,                 // Integral Constant: fixed point 0.4
    input       [3:0]   kd,                 // Derivative Constant: fixed point 0.4

    output reg  [15:0]  debug_signals,  
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
reg     [2:0]       curr_step;
wire    [63:0]      proportional_error;             // The proportional instantenous error
wire    [31:0]      integral_error;                 // The cumulative error
wire    [31:0]      derivative_error;               // The derivative error
reg     [15:0]      elapsed_time;                   // The elapsed time since the last update
wire    [15:0]      delta_12p4;
reg     [15:0]      last_delta_12p4;                // The last error from the PID controller
wire    [15:0]      target_12p4, current_12p4;      // Fixed point 12.4 format
wire    [11:0]      delta_angle;                    // The angle difference between the target and current angle
wire                calc_updated;                   // Delta angle has been updated
reg                 rd_done, i2c_rd_done_ff;        // Read done
wire    [11:0]      ratio_int;
wire    [7:0]       debug_ratio; 
wire    [7:0]       profile_coeff   [7:0];          // Acceleration profile coefficents

assign profile_coeff[0][7:0] = profile[1*8-1:0*8];
assign profile_coeff[1][7:0] = profile[2*8-1:1*8];
assign profile_coeff[2][7:0] = profile[3*8-1:2*8];
assign profile_coeff[3][7:0] = profile[4*8-1:3*8];
assign profile_coeff[4][7:0] = profile[5*8-1:4*8];
assign profile_coeff[5][7:0] = profile[6*8-1:5*8];
assign profile_coeff[6][7:0] = profile[7*8-1:6*8];
assign profile_coeff[7][7:0] = profile[8*8-1:7*8];

assign target_12p4  = target_angle  << 4;
assign current_12p4 = current_angle << 4;
assign delta_12p4   = delta_angle   << 4;

assign debug_ratio = ratio_int >> 4;

always @(negedge reset_n or posedge clock) begin
    if(~reset_n) begin
        angle_done          <= 1'b0; 
        pwm_update          <= 1'b0; 
        startup_fail        <= 1'b0; 
        debug_signals       <= 16'hDEAD;
        elapsed_time        <= 16'b1;
        last_delta_12p4     <= 12'b0;
        rd_done             <= 1'b0;
        i2c_rd_done_ff      <= 1'b0;
        state               <= IDLE;
        curr_step           <= 3'b0;
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

            end

            ACCEL: begin
                if(rd_done) begin
                    if(curr_step <= 3'b111) begin
                        case(curr_step)
                            3'b000, 3'b001, 3'b010: pwm_ratio   <= ratio_int >> 8;
                            3'b011, 3'b100: pwm_ratio           <= ratio_int >> 7;
                            3'b101, 3'b110: pwm_ratio           <= ratio_int >> 6;
                            default:        pwm_ratio           <= ratio_int >> 5;   
                        endcase
                    end

                    if(curr_step < 3'b111) 
                        curr_step <= curr_step + 3'b1;    
                    else
                        state <= CRUISE;
                end
            end 

            CRUISE: begin
                if(delta_angle < 12'd50)
                    state <= DECEL;
                
                pwm_ratio <= ratio_int >> 4;
            end 

            DECEL: begin
                if(delta_angle < 12'd10) begin
                    state       <= IDLE;
                    angle_done  <= 1'b1;
                end
            end 

            default: 
                state <= IDLE;

        endcase


        if (rd_done) begin
            elapsed_time            <= elapsed_time + 16'b1;
            last_delta_12p4         <= delta_12p4;
        end 
    end
end

assign proportional_error       = kp * delta_12p4;
assign integral_error           = ki * (delta_12p4 >> 8) * (elapsed_time >> 8);
assign derivative_error         = kd * (delta_12p4 - last_delta_12p4) / elapsed_time;
assign ratio_int                = (proportional_error + integral_error + derivative_error) >> 4;

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