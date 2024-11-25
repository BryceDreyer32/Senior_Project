// Copyright 2024
// Bryce's Senior Project
// Description: This is the RTL for the I2C which communicates between the PWM controller and the encoder.
module i2c(
    input               reset_n,        // Active low reset
    input               clock,          // The main clock
    input               angle_done,     // Whether or not we are at the target angle
    output      [11:0]  raw_angle,      // The raw angle from the AS5600            
    output              scl,            // The I2C clock
    inout               sda             // The I2C bi-directional data
);

localparam      ENCODER_ADDRESS     = 7'h36;
localparam      CLOCK_DIV_RATIO     = 6'h32;

localparam      IDLE        = 4'h0;
localparam      START       = 4'h1;
localparam      DEV_WR      = 4'h2;
localparam      ACK0        = 4'h3;
localparam      READ_ADDR   = 4'h4;
localparam      ACK1        = 4'h5;
localparam      REP_START   = 4'h6;
localparam      DEV_RD      = 4'h7;
localparam      ACK2        = 4'h8;
localparam      DATA0       = 4'h9;
localparam      ACK3        = 4'hA;
localparam      DATA1       = 4'hB;
localparam      NACK        = 4'hC;
localparam      STOP        = 4'hD;
localparam      PAUSE       = 4'hE;

localparam      WRITE_DEVICE    = 8'h6C;
localparam      READ_DEVICE     = 8'h6D;
localparam      READ_RAW_ANGLE  = 8'h0C;

reg     [2:0]   bit_counter;
reg     [3:0]   ns, ps;
reg     [1:0]   substate;
wire            PHASE1, PHASE2, PHASE3, PHASE4;
reg             scl_int, sda_int, tri_mode;
reg             wr_bit, rd_bit;
reg     [7:0]   read_data0, read_data1;

always @(posedge clock or negedge reset_n) begin
    if(~reset_n) begin
        substate[1:0]   <= 2'h0;
        wr_bit          <= 1'h0;
        read_data0[7:0] <= 8'b0;
        read_data1[7:0] <= 8'b0;
    end
    else begin
        substate[1:0]   <= substate[1:0] + 2'h1;
        if(ps == DEV_WR)
            wr_bit        <= WRITE_DEVICE[8'd7 - bit_counter];
        else if(ps == READ_ADDR)
            wr_bit        <= READ_RAW_ANGLE[8'd7 - bit_counter];
        else if(ps == DEV_RD)
            wr_bit        <= READ_DEVICE[8'd7 - bit_counter];
        else
            wr_bit    <= 1'bx;

        if((ps == DATA0) & PHASE3)
            read_data1[8'd7 - bit_counter] = rd_bit;
        
        else if((ps == DATA1) & PHASE3)
            read_data0[8'd7 - bit_counter] = rd_bit;
    end
end

assign PHASE1 = (substate[1:0] == 2'b00);
assign PHASE2 = (substate[1:0] == 2'b01);
assign PHASE3 = (substate[1:0] == 2'b10);
assign PHASE4 = (substate[1:0] == 2'b11);

always @(posedge PHASE1 or negedge reset_n) begin
    if(~reset_n) begin
        ps              <= IDLE;
        bit_counter     <= 3'b0;        
    end
    else begin
        ps <= ns;        
        if((ps == DEV_WR) | (ps == READ_ADDR) | (ps == DEV_RD) | (ps == DATA0) | (ps == DATA1))
            bit_counter <= bit_counter + 3'h1;
        else
            bit_counter <= 3'h0;        
    end
end

always @(*) begin
    case(ps)
        IDLE: begin
            if(~angle_done)
                ns = START;
            else
                ns = IDLE;
        end

        START: begin
            ns = DEV_WR;
        end

        DEV_WR: begin
            if(bit_counter == 4'h7)
                ns = ACK0;
            else
                ns = DEV_WR;
        end

        ACK0: begin
            ns = READ_ADDR;
        end

        READ_ADDR: begin
            if(bit_counter == 4'h7)
                ns = ACK1;
            else
                ns = READ_ADDR;
        end

        ACK1: begin
            ns = REP_START;
        end


        REP_START: begin
            ns = DEV_RD;
        end

        DEV_RD: begin
            if(bit_counter == 4'h7)
                ns = ACK2;
            else
                ns = DEV_RD; 
        end

        ACK2: begin
            ns = DATA0;
        end

        DATA0: begin
            if(bit_counter == 4'h7)
                ns = ACK3;
            else
                ns = DATA0;
        end

        ACK3: begin
            ns = DATA1;
        end

        DATA1: begin
            if(bit_counter == 4'h7)
                ns = NACK;
            else 
                ns = DATA1;
        end

        NACK: begin
            ns = STOP;
        end

        STOP: begin
            ns = PAUSE;
        end

        PAUSE: begin
            if(angle_done)
                ns = IDLE;
            else
                ns = START;
        end

        default: begin
            ns = IDLE;
        end
    endcase
end

// I2C Clock and Data driver
always @(*) begin
    scl_int     = 1'b1;
    sda_int     = 1'b1;
    tri_mode    = 1'b0;

    case(ps)
        IDLE: begin
            scl_int = 1'b1;
            sda_int = 1'b1;
        end

        START: begin
            //  A high-to-low transition on the SDA line while the SCL is high defines a START condition
            tri_mode    = 1'b0;
            if(PHASE1) begin
                scl_int = 1'b1;
                sda_int = 1'b1;
            end 
            else if(PHASE2) begin
                scl_int = 1'b1;
                sda_int = 1'b0;
            end
             else if(PHASE3) begin
                scl_int = 1'b1;
                sda_int = 1'b0;
            end
             else if(PHASE4) begin
                scl_int = 1'b0;
                sda_int = 1'b0;
            end
        end

        DEV_WR: begin
            tri_mode    = 1'b0;
            if(PHASE1) begin
                scl_int = 1'b0;
                sda_int = wr_bit;
            end 
            else if(PHASE2) begin
                scl_int = 1'b0;
                sda_int = wr_bit;
            end
             else if(PHASE3) begin
                scl_int = 1'b1;
                sda_int = wr_bit;
            end
             else if(PHASE4) begin
                scl_int = 1'b1;
                sda_int = wr_bit;
            end
        end

        ACK0: begin
            tri_mode = 1'b1;
            if(PHASE1) begin
                scl_int = 1'b0;
                sda_int = 1'bZ;
            end 
            else if(PHASE2) begin
                scl_int = 1'b0;
                sda_int = 1'bZ;
            end
             else if(PHASE3) begin
                scl_int = 1'b1;
                sda_int = 1'bZ;
            end
             else if(PHASE4) begin
                scl_int = 1'b1;
                sda_int = 1'bZ;
            end
        end

        READ_ADDR: begin
            // The SDA is x during the 1st phase because the address calculation requires that we flop it (verilog issue)
            // But its value is stable before the clock rise.
            tri_mode    = 1'b0;
            if(PHASE1) begin
                scl_int = 1'b0;
                sda_int = wr_bit;
            end 
            else if(PHASE2) begin
                scl_int = 1'b0;
                sda_int = wr_bit;
            end
             else if(PHASE3) begin
                scl_int = 1'b1;
                sda_int = wr_bit;
            end
             else if(PHASE4) begin
                scl_int = 1'b1;
                sda_int = wr_bit;
            end
        end

        ACK1: begin
            tri_mode = 1'b1;
            if(PHASE1) begin
                scl_int = 1'b0;
                sda_int = 1'bZ;
            end 
            else if(PHASE2) begin
                scl_int = 1'b0;
                sda_int = 1'bZ;
            end
             else if(PHASE3) begin
                scl_int = 1'b1;
                sda_int = 1'bZ;
            end
             else if(PHASE4) begin
                scl_int = 1'b1;
                sda_int = 1'bZ;
            end
        end

        REP_START: begin
            tri_mode    = 1'b0;
            if(PHASE1) begin
                scl_int = 1'b0;
                sda_int = 1'b0;
            end 
            else if(PHASE2) begin
                scl_int = 1'b0;
                sda_int = 1'b1;
            end
             else if(PHASE3) begin
                scl_int = 1'b1;
                sda_int = 1'b1;
            end
             else if(PHASE4) begin
                scl_int = 1'b1;
                sda_int = 1'b0;
            end
        end

        DEV_RD: begin
            tri_mode    = 1'b0;
            if(PHASE1) begin
                scl_int = 1'b0;
                sda_int = wr_bit;
            end 
            else if(PHASE2) begin
                scl_int = 1'b0;
                sda_int = wr_bit;
            end
             else if(PHASE3) begin
                scl_int = 1'b1;
                sda_int = wr_bit;
            end
             else if(PHASE4) begin
                scl_int = 1'b1;
                sda_int = wr_bit;
            end
        end

        ACK2: begin
            tri_mode = 1'b1;
            if(PHASE1) begin
                scl_int = 1'b0;
                sda_int = 1'bZ;
            end 
            else if(PHASE2) begin
                scl_int = 1'b0;
                sda_int = 1'bZ;
            end
             else if(PHASE3) begin
                scl_int = 1'b1;
                sda_int = 1'bZ;
            end
             else if(PHASE4) begin
                scl_int = 1'b1;
                sda_int = 1'bZ;
            end
        end

        DATA0: begin
            tri_mode = 1'b1;
            if(PHASE1) begin
                scl_int = 1'b0;
                sda_int = 1'bZ;
            end 
            else if(PHASE2) begin
                scl_int = 1'b0;
                sda_int = 1'bZ;
            end
             else if(PHASE3) begin
                scl_int = 1'b1;
                sda_int = 1'bZ;
            end
             else if(PHASE4) begin
                scl_int = 1'b1;
                sda_int = 1'bZ;
                rd_bit  = sda;
            end
        end

        ACK3: begin
            // When reading data the controller has to pull down the SDA to ACK
            tri_mode    = 1'b0;
            if(PHASE1) begin
                scl_int = 1'b0;
                sda_int = 1'b0;
            end 
            else if(PHASE2) begin
                scl_int = 1'b0;
                sda_int = 1'b0;
            end
             else if(PHASE3) begin
                scl_int = 1'b1;
                sda_int = 1'b0;
            end
             else if(PHASE4) begin
                scl_int = 1'b1;
                sda_int = 1'b0;
            end
        end

        DATA1: begin
            tri_mode = 1'b1;
            if(PHASE1) begin
                scl_int = 1'b0;
                sda_int = 1'bZ;
            end 
            else if(PHASE2) begin
                scl_int = 1'b0;
                sda_int = 1'bZ;
            end
             else if(PHASE3) begin
                scl_int = 1'b1;
                sda_int = 1'bZ;
            end
             else if(PHASE4) begin
                scl_int = 1'b1;
                sda_int = 1'bZ;
                rd_bit  = sda;
            end
        end

        NACK: begin
            // In the NACK condition, the controller will drive the SDA to 1
            tri_mode    = 1'b0;
            if(PHASE1) begin
                scl_int = 1'b0;
                sda_int = 1'b1;
            end 
            else if(PHASE2) begin
                scl_int = 1'b0;
                sda_int = 1'b1;
            end
             else if(PHASE3) begin
                scl_int = 1'b1;
                sda_int = 1'b1;
            end
             else if(PHASE4) begin
                scl_int = 1'b1;
                sda_int = 1'b1;
            end   
        end

        STOP: begin
            // The STOP condition occurs after a NACK, we keep the SDA at 1 for the first phase, then pull it low for the 2nd
            // and 3rd phases, then it is pulled high for the 4th phase.
            //A low-to-high transition on the SDA line while the SCL is high defines a STOP condition.
            tri_mode    = 1'b0;
            if(PHASE1) begin
                scl_int = 1'b0;
                sda_int = 1'b1;
            end 
            else if(PHASE2) begin
                scl_int = 1'b0;
                sda_int = 1'b0;
            end
             else if(PHASE3) begin
                scl_int = 1'b1;
                sda_int = 1'b0;
            end
             else if(PHASE4) begin
                scl_int = 1'b1;
                sda_int = 1'b1;
            end 
        end

        PAUSE: begin
            tri_mode= 1'b0;
            scl_int = 1'b1;
            sda_int = 1'b1;
        end

        default: begin
            tri_mode= 1'b0;
            scl_int = 1'b1;
            sda_int = 1'b1;
        end
    endcase    
end

assign scl = scl_int;
assign sda = tri_mode ? 1'bZ : sda_int;
assign raw_angle = {read_data1[11:8], read_data0[7:0]};
endmodule