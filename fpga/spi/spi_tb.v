// Copyright 2024
// Bryce's Senior Project
// Description: This is the TestBench for SPI

module test();
    reg             clock       = 0;
    reg             spi_clk     = 0;
    reg             reset_n     = 1;
    reg             cs_n        = 1;
//    wire           reset_n;   	     // Active low reset
//    wire           clock;     	     // The main clock
    wire   [5:0]    address;   	     // Read / write address
    wire            write_en;  	     // Write enable
    wire   [7:0]    wr_data;   	     // Write data
    wire            read_en;  	     // Read enable
    wire  [7:0]     rd_data;   	     // Read data

    wire          fault0;    	     // Fault signal from motor
    wire  [6:0]   adc_temp0; 	     // Adc temperature from motor
    wire          fault1;    	     // Fault signal from motor
    wire  [6:0]   adc_temp1; 	     // Adc temperature from motor
    wire          fault2;    	     // Fault signal from motor
    wire  [6:0]   adc_temp2; 	     // Adc temperature from motor
    wire          fault3;    	     // Fault signal from motor
    wire  [6:0]   adc_temp3; 	     // Adc temperature from motor
    wire          fault4;    	     // Fault signal from motor
    wire  [6:0]   adc_temp4; 	     // Adc temperature from motor
    wire          fault5;    	     // Fault signal from motor
    wire  [6:0]   adc_temp5; 	     // Adc temperature from motor
    wire          fault6;    	     // Fault signal from motor
    wire  [6:0]   adc_temp6; 	     // Adc temperature from motor
    wire          fault7;    	     // Fault signal from motor
    wire  [6:0]   adc_temp7; 	     // Adc temperature from motor

    wire          brake0;    	     // Brake control
    wire          enable0;   	     // Motor enable
    wire          direction0;	     // Motor direction
    wire  [4:0]   pwm0;      	     // PWM control  
    wire          brake1;    	     // Brake control
    wire          enable1;   	     // Motor enable
    wire          direction1;	     // Motor direction
    wire  [4:0]   pwm1;      	     // PWM control  
    wire          brake2;    	     // Brake control
    wire          enable2;   	     // Motor enable
    wire          direction2;	     // Motor direction
    wire  [4:0]   pwm2;      	     // PWM control  
    wire          brake3;    	     // Brake control
    wire          enable3;   	     // Motor enable
    wire          direction3;	     // Motor direction
    wire  [4:0]   pwm3;      	     // PWM control
    wire          brake4;    	     // Brake control
    wire          enable4;   	     // Motor enable
    wire          direction4;	     // Motor direction
    wire  [4:0]   pwm4;      	     // PWM control  
    wire          brake5;    	     // Brake control
    wire          enable5;   	     // Motor enable
    wire          direction5;	     // Motor direction
    wire  [4:0]   pwm5;      	     // PWM control  
    wire          brake6;    	     // Brake control
    wire          enable6;   	     // Motor enable
    wire          direction6;	     // Motor direction
    wire  [4:0]   pwm6;      	     // PWM control  
    wire          brake7;    	     // Brake control
    wire          enable7;   	     // Motor enable
    wire          direction7;	     // Motor direction
    wire  [4:0]   pwm7;      	     // PWM control  

    wire  [7:0]   target_angle0;   // Rotation target angle
    wire  [7:0]   current_angle0;  // The current angle
    wire  [7:0]   target_angle1;   // Rotation target angle
    wire  [7:0]   current_angle1;  // The current angle
    wire  [7:0]   target_angle2;   // Rotation target angle
    wire  [7:0]   current_angle2;  // The current angle
    wire  [7:0]   target_angle3;   // Rotation target angle
    wire  [7:0]   current_angle3;  // The current angle

    wire  [7:0]   servo_position0; // Servo 0 target position
    wire  [7:0]   servo_position1; // Servo 1 target position
    wire  [7:0]   servo_position2; // Servo 2 target position
    wire  [7:0]   servo_position3; // Servo 3 target position

    reg [15:0]      test_mosi   = 0;
    reg [7:0]       test_data   = 8'h56;
    reg [5:0]       test_addr   = 8'h4;
    reg             test_parity = 0;
    reg             test_rw     = 1;

always
    #1  clock = ~clock;

always 
    #3 spi_clk = ~spi_clk;


initial begin
    $display("Starting SPI test");
    #1 reset_n = 1;
    #1 reset_n = 0;

    $display("Deasserting reset");
    #8 reset_n = 1;

    $display("--- WRITE TEST ---");
    $display("Asserting cs_n");
    test_rw = 0;
    test_mosi = {1'b0, 1'b1, test_addr[5:0], test_data[7:0]};
    #12 cs_n = 0;

    $display("Running for 16 spiclk cycles");
    #(16*3*2);

    $display("Deasserting cs_n");
    #5 cs_n = 1;

    #(10*3*2);

    // For the read, we first send 1 byte which contains the read flag, parity, and read address
    $display("--- READ TEST ---");
    $display("Asserting cs_n, writing the address to be read");
    test_rw = 1;
    test_mosi = {1'b1, 1'b0, test_addr[5:0], 8'hFF};
    #2 cs_n = 0;

    $display("Running for 16 spiclk cycles");
    #(16*3*2);    

    $display("Deasserting cs_n");
    #3 cs_n = 1;

    #(5*3*2);

    // Then we read the data byte back on the MISO (while MOSI is held high the whole time)
    $display("Asserting cs_n, and reading back data read");
    test_rw = 1;
    test_mosi = 16'hFFFF;
    #2 cs_n = 0;

    $display("Running for 16 spiclk cycles");
    #(16*3*2);    

    $display("Deasserting cs_n");
    #3 cs_n = 1;


    #(50*3*2) $finish;
end

reg clk_gate = 1;
wire gate_spi_clk;
always @(negedge spi_clk or negedge reset_n) begin
    if(~reset_n)
        clk_gate    <= 1'b0;
    else begin
        clk_gate    <= ~cs_n;
    end
end
assign gate_spi_clk = spi_clk & clk_gate & ~cs_n;

reg cs_n_ff, cs_n_ff2;

always @(negedge gate_spi_clk) begin
    test_mosi[15:0] <= {test_mosi[14:0], 1'b0};
end
assign mosi = test_mosi[15];

always @(*) begin
    if(test_rw) begin
        test_addr   = 6'h4;
        test_rw     = 1'h1;
        test_parity = ^({test_rw, test_addr[5:0], test_data[7:0]});        
    end
    else begin
        test_addr   = 6'h4;
        test_rw     = 1'h0;
        test_parity = ^({test_rw, test_addr[5:0], test_data[7:0]});                
    end
end


spi dut(
	.reset_n            (reset_n),          // Active low reset
	.clock              (clock),            // The main clock
	.spi_clk            (gate_spi_clk),     // The SPI clock
	.cs_n               (cs_n),             // Active low chip select
	.mosi               (mosi),             // Master out slave in
	.miso               (miso),             // Master in slave out (SPI mode 0)
    .address            (address),   	    // Read / write address
    .write_en           (write_en),  	    // Write enable
    .wr_data            (wr_data),   	    // Write data
    .read_en            (read_en),   	    // Read enable
    .rd_data            (rd_data)   	    // Read data);
);

reg_file rf(
    .reset_n            (reset_n),   	    // Active low reset
    .clock              (clock),     	    // The main clock
    .address            (address),   	    // Read / write address
    .write_en           (write_en),  	    // Write enable
    .wr_data            (wr_data),   	    // Write data
    .read_en            (read_en),   	    // Read enable
    .rd_data            (rd_data),   	    // Read data
				     
    .fault0             (fault0),    	    // Fault signal from motor
    .adc_temp0          (adc_temp0), 	    // Adc temperature from motor
    .fault1             (fault1),    	    // Fault signal from motor
    .adc_temp1          (adc_temp1), 	    // Adc temperature from motor
    .fault2             (fault2),    	    // Fault signal from motor
    .adc_temp2          (adc_temp2), 	    // Adc temperature from motor
    .fault3             (fault3),    	    // Fault signal from motor
    .adc_temp3          (adc_temp3), 	    // Adc temperature from motor
    .fault4             (fault4),    	    // Fault signal from motor
    .adc_temp4          (adc_temp4), 	    // Adc temperature from motor
    .fault5             (fault5),    	    // Fault signal from motor
    .adc_temp5          (adc_temp5), 	    // Adc temperature from motor
    .fault6             (fault6),    	    // Fault signal from motor
    .adc_temp6          (adc_temp6), 	    // Adc temperature from motor
    .fault7             (fault7),    	    // Fault signal from motor
    .adc_temp7          (adc_temp7), 	    // Adc temperature from motor
				     
	.brake0             (/*brake0*/),    	    // Brake control
    .enable0            (/*enable0*/),   	    // Motor enable
    .direction0         (/*direction0*/),	    // Motor direction
    .pwm0               (/*pwm0*/),      	    // PWM control  
    .brake1             (/*brake1*/),    	    // Brake control
    .enable1            (/*enable1*/),   	    // Motor enable
    .direction1         (/*direction1*/),	    // Motor direction
    .pwm1               (/*pwm1*/),      	    // PWM control  
    .brake2             (/*brake2*/),    	    // Brake control
    .enable2            (/*enable2*/),   	    // Motor enable
    .direction2         (/*direction2*/),	    // Motor direction
    .pwm2               (/*pwm2*/),      	    // PWM control  
    .brake3             (/*brake3*/),    	    // Brake control
    .enable3            (/*enable3*/),   	    // Motor enable
    .direction3         (/*direction3*/),	    // Motor direction
    .pwm3               (/*pwm3*/),      	    // PWM control
    .brake4             (/*brake4*/),    	    // Brake control
    .enable4            (/*enable4*/),   	    // Motor enable
    .direction4         (/*direction4*/),	    // Motor direction
    .pwm4               (/*pwm4*/),      	    // PWM control  
    .brake5             (/*brake5*/),    	    // Brake control
    .enable5            (/*enable5*/),   	    // Motor enable
    .direction5         (/*direction5*/),	    // Motor direction
    .pwm5               (/*pwm5*/),      	    // PWM control  
    .brake6             (/*brake6*/),    	    // Brake control
    .enable6            (/*enable6*/),   	    // Motor enable
    .direction6         (/*direction6*/),	    // Motor direction
    .pwm6               (/*pwm6*/),      	    // PWM control  
    .brake7             (/*brake7*/),    	    // Brake control
    .enable7            (/*enable7*/),   	    // Motor enable
    .direction7         (/*direction7*/),	    // Motor direction
    .pwm7               (/*pwm7*/),      	    // PWM control  
					 
    .target_angle0      (target_angle0),    // Rotation target angle
    .current_angle0     (current_angle0),   // The current angle
    .target_angle1      (target_angle1),    // Rotation target angle
    .current_angle1     (current_angle1),   // The current angle
    .target_angle2      (target_angle2),    // Rotation target angle
    .current_angle2     (current_angle2),   // The current angle
    .target_angle3      (target_angle3),    // Rotation target angle
    .current_angle3     (current_angle3),   // The current angle
	
    .servo_position0    (servo_position0), // Servo 0 target position
    .servo_position1    (servo_position1), // Servo 1 target position
    .servo_position2    (servo_position2), // Servo 2 target position
    .servo_position3    (servo_position3)  // Servo 3 target position
);

assign fault0 = 0;
assign fault1 = 0;
assign fault2 = 0;
assign fault3 = 0;
assign adc_temp0 = 'hA;
assign adc_temp1 = 'hA;
assign adc_temp2 = 'hA;
assign adc_temp3 = 'hA;
assign current_angle0 = 'h55;
assign current_angle1 = 'h55;
assign current_angle2 = 'h55;
assign current_angle3 = 'h55;



initial begin
    $dumpfile("spi.vcd");
    $dumpvars(0,test);
  end

endmodule