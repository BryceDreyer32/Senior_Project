// Copyright 2024
// Bryce's Senior Project
// Description: This is the RTL for Address Decoder Logic

module addr_dec ( 
    input           reset_n,            // Active low reset
    input           spi_clock,          // The SPI clock
    input           fpga_clock,         // The FPGA clock
    input           cs_n,               // Active low chip select to the SPI receiver, used by the address decoder to determine when SPI data is ready
    input [15:0]    spi_out,            // The output of the SPI
    input [11:0]    pwm_done,           // Updated PWM ratio has been applied (1 cycle long pulse)
    output [7:0]    sr0_pwm_target,     // Swerve rotation PWM 0
    output [7:0]    sr1_pwm_target,     // Swerve rotation PWM 1
    output [7:0]    sr2_pwm_target,     // Swerve rotation PWM 2
    output [7:0]    sr3_pwm_target,     // Swerve rotation PWM 3
    output [7:0]    sd0_pwm_target,     // Swerve drive PWM 0
    output [7:0]    sd1_pwm_target,     // Swerve drive PWM 1
    output [7:0]    sd2_pwm_target,     // Swerve drive PWM 2
    output [7:0]    sd3_pwm_target,     // Swerve drivePWM 3
    output [7:0]    servo0_pwm_target,  // Arm servo PWM 0
    output [7:0]    servo1_pwm_target,  // Arm servo PWM 1
    output [7:0]    servo2_pwm_target,  // Arm servo PWM 2
    output [7:0]    servo3_pwm_target,  // Arm servo PWM 3
    output [11:0]   pwm_update,         // Indicator to update PWM ratio
    output reg      crc_error           // Indication that a CRC error has been detected
);

reg cs_n_ff, cs_n_ff2, cs_n_ff3;
reg data_ready;
reg [11:0] update;
wire [3:0]  address;
wire [3:0] crc;

// cs_n clock crossing
always @(posedge spi_clock or negedge reset_n) begin
    if(~reset_n)
        cs_n_ff <= 1'b0;
    else
        cs_n_ff <= cs_n;
end

always @(posedge fpga_clock or negedge reset_n) begin
    if(~reset_n) begin
        cs_n_ff2   <= 1'b0;
        cs_n_ff3   <= 1'b0;
        data_ready <= 1'b0;
        update     <= 12'b0;
        crc_error  <= 1'b0;
    end
    else begin
        cs_n_ff2   <= cs_n_ff;
        cs_n_ff3   <= cs_n_ff2;
        data_ready <= cs_n_ff2 & ~cs_n_ff3;

        
        if(data_ready) begin
            // Update indicator for the Swerve Rotation
            update[0] <= ((address[3:0] == 4'h0) | (address[3:0] == 4'h1) | (address[3:0] == 4'h4));
            update[1] <= ((address[3:0] == 4'h0) | (address[3:0] == 4'h1) | (address[3:0] == 4'h5));
            update[2] <= ((address[3:0] == 4'h0) | (address[3:0] == 4'h1) | (address[3:0] == 4'h6));
            update[3] <= ((address[3:0] == 4'h0) | (address[3:0] == 4'h1) | (address[3:0] == 4'h7));

            // Update indicator for the Swerve Drive
            update[4] <= ((address[3:0] == 4'h0) | (address[3:0] == 4'h2) | (address[3:0] == 4'h8));
            update[5] <= ((address[3:0] == 4'h0) | (address[3:0] == 4'h2) | (address[3:0] == 4'h9));
            update[6] <= ((address[3:0] == 4'h0) | (address[3:0] == 4'h2) | (address[3:0] == 4'hA));
            update[7] <= ((address[3:0] == 4'h0) | (address[3:0] == 4'h2) | (address[3:0] == 4'hB));

            // Update indicator for the Arm Servos
            update[8]  <= ((address[3:0] == 4'h0) | (address[3:0] == 4'hC));
            update[9]  <= ((address[3:0] == 4'h0) | (address[3:0] == 4'hD));
            update[10] <= ((address[3:0] == 4'h0) | (address[3:0] == 4'hE));
            update[11] <= ((address[3:0] == 4'h0) | (address[3:0] == 4'hF));
        end 
        else begin
            update[0]  <= pwm_done[0]  ? 1'b0 : update[0];
            update[1]  <= pwm_done[1]  ? 1'b0 : update[1];
            update[2]  <= pwm_done[2]  ? 1'b0 : update[2];
            update[3]  <= pwm_done[3]  ? 1'b0 : update[3];
            update[4]  <= pwm_done[4]  ? 1'b0 : update[4];
            update[5]  <= pwm_done[5]  ? 1'b0 : update[5];
            update[6]  <= pwm_done[6]  ? 1'b0 : update[6];
            update[7]  <= pwm_done[7]  ? 1'b0 : update[7];
            update[8]  <= pwm_done[8]  ? 1'b0 : update[8];
            update[9]  <= pwm_done[9]  ? 1'b0 : update[9];
            update[10] <= pwm_done[10] ? 1'b0 : update[10];
            update[11] <= pwm_done[11] ? 1'b0 : update[11];
        end


        crc_error <= |(crc[3:0] ^ spi_out[15:12]);
    end
end

// CRC calculation (not real, just practice version)
assign crc[3:0] = {(spi_out[11] ^ spi_out[7] ^ spi_out[5] ^ spi_out[3]), (spi_out[10] ^ spi_out[5] ^ spi_out[3] ^ spi_out[1]), (spi_out[9] ^ spi_out[6] ^ spi_out[4] ^ spi_out[2]), (spi_out[8] ^ spi_out[4] ^ spi_out[2] ^ spi_out[0])};

// If there is a CRC error set the address to the garbage disposal value 4'h3 otherwise, pass the SPI address through
assign address[3:0] = crc_error ? 4'h3 : spi_out[11:8]; 

// Assign the SPI data to each of the targets
assign sr0_pwm_target     = spi_out[7:0];
assign sr1_pwm_target     = spi_out[7:0];
assign sr2_pwm_target     = spi_out[7:0];
assign sr3_pwm_target     = spi_out[7:0];
assign sd0_pwm_target     = spi_out[7:0];
assign sd1_pwm_target     = spi_out[7:0];
assign sd2_pwm_target     = spi_out[7:0];
assign sd3_pwm_target     = spi_out[7:0];
assign servo0_pwm_target  = spi_out[7:0];
assign servo1_pwm_target  = spi_out[7:0];
assign servo2_pwm_target  = spi_out[7:0];
assign servo3_pwm_target  = spi_out[7:0];

assign pwm_update[11:0] = update[11:0];

endmodule