// Copyright 2024
// Bryce's Senior Project
// Description: This is the RTL for PWM Control

module pwm_ctrl(
    input               reset_n,        // Active low reset
    input               clock,          // The main clock

    //FPGA Subsystem Interface
    input   [7:0]       target_angle,   // The angle the wheel needs to move to in degrees. This number is multiplied by 2 internally
    input               angle_update,   // Signals when an angle update is available
    output              angle_done,     // Output sent when angle has been adjusted to target_angle

    //PWM Interface
    input               pwm_done,       // Updated PWM ratio has been applied (1 cycle long pulse)
    output              pwm_enable,     // Enables the PWM output
    output  [7:0]       pwm_ratio,      // The high-time of the PWM signal out of 255.
    output              pwm_update,     // Request an update to the PWM ratio

    //I2C Interface
    output              sck,            // The I2C clock
    inout               sda             // The I2C bi-directional data
);

localparam      IDLE        = 4'b0;
localparam      START       = 4'd1;
localparam      DEV_ADDR0   = 4'd2;
localparam      WORD_ADDR   = 4'd3;
localparam      DEV_ADDR1   = 4'd4;
localparam      RD_DATA0    = 4'd5;
localparam      RD_DATA1    = 4'd6;
localparam      NACK        = 4'd7;
localparam      STOP        = 4'd8;
localparam      PAUSE       = 4'd9;
localparam      ERROR       = 4'd15;

reg     [1:0]       instruction;
reg     [7:0]       wr_data;
wire    [7:0]       rd_data;
wire                i2c_done, isSending, sdaOutReg;
reg     [3:0]       ps, ns;
reg                 send_nack;
reg                 enable;
reg     [15:0]      counter;

// FSM to control the pwm ctrl
always @(posedge clock or negedge reset_n)  begin
    if(~reset_n) begin
        ps <= IDLE;
        counter <= 16'b0;
    end

    else begin
        ps <= ns;
        if(ps == PAUSE)
            counter <= counter + 16'b1;
        else
            counter <= 16'b0;
    end
end

always @(*) begin
    instruction = 3'b0;
    wr_data     = 8'b0;
    send_nack   = 1'b0;
    enable      = 1'b1;

    case(ps)
        IDLE: begin
            enable = 1'b0;
            if(angle_update)
                ns = START;
        end

        START: begin
            instruction = 3'd0;
            if(i2c_done)
                ns = DEV_ADDR0;
            else
                ns = START;
        end

        DEV_ADDR0: begin
            instruction = 3'd3;
            wr_data = {7'h36, 1'h0};
            if(i2c_done)
                ns = WORD_ADDR;
            else
                ns = DEV_ADDR0;
        end

        WORD_ADDR: begin
            instruction = 3'd3;
            wr_data = 8'hC;
            if(i2c_done)
                ns = DEV_ADDR1;
            else
                ns = WORD_ADDR;
        end

        DEV_ADDR1: begin
            instruction = 3'd3;
            wr_data = {7'h36, 1'h0};
            if(i2c_done)
                ns = RD_DATA0;
            else
                ns = DEV_ADDR1;
        end

        RD_DATA0: begin
            instruction = 3'd2;
            if(i2c_done)
                ns = RD_DATA1;
            else
                ns = RD_DATA0;
        end

        RD_DATA1: begin
            instruction = 3'd2;
            if(i2c_done)
                ns = NACK;
            else
                ns = RD_DATA1;
        end

        NACK: begin
            send_nack = 1'b1;
            if(i2c_done)
                ns = STOP;
            else
                ns = NACK;
        end

        STOP: begin
            instruction = 3'd1;
            if(i2c_done)
                ns = PAUSE;
            else
                ns = STOP;
        end

        PAUSE: begin
           /*  if(angle_done)
                ns = IDLE;
            else
                ns = START;  */
            instruction = 3'd1;
            if(counter == 16'h3FF)
                ns = START;
            else begin
                ns = PAUSE;
                enable = 1'b0;
            end
        end

        default: begin
            ns = IDLE;
        end
    endcase
end
    

angle_to_pwm a_to_pwm(
    .reset_n        (reset_n),  	    // Active low reset
    .clock          (clock),	        // The main clock
    .target_angle   (target_angle),     // The angle the wheel needs to move to in degrees. This number is multiplied by 2 internally
    .current_angle  (rd_data[7:0]),    // The angle read from the motor encoder
    .pwm_done       (pwm_done),         // Indicator from PWM that the pwm_ratio has been applied
    .angle_update   (angle_update),     // Request to update the angle
    .angle_done     (angle_done),       // Indicator that the angle has been applied 
    .pwm_enable     (pwm_enable),
    .pwm_update     (pwm_update),       // Request an update to the PWM ratio
    .pwm_ratio      (pwm_ratio),        // The high-time of the PWM signal out of 255.
    .pwm_direction  ()                  // The direction of the motor
);

i2c i2c(
    .clk            (clock),
    .reset_n        (reset_n),
    .sdaIn          (sda),
    .sdaOutReg      (sdaOutReg),
    .isSending      (isSending),
    .scl            (sck),
    .instruction    (instruction[1:0]),
    .enable         (enable),
    .byteToSend     (wr_data[7:0]),
    .send_nack      (send_nack),
    .byteReceived   (rd_data[7:0]),
    .complete       (i2c_done)
);
 
assign sda = isSending ? sdaOutReg : 1'bz;

endmodule