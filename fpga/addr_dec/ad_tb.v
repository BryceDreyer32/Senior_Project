module test (); 

reg		    	reset_n		= 0;
reg			    spi_clock	= 0;
reg		    	fpga_clock	= 0;
reg			    cs_n		= 1;
reg	 [15:0]	spi_out		= 0;
reg	 [11:0]	pwm_done	= 0;
wire [7:0]	sr0_pwm_target;
wire [7:0]	sr1_pwm_target;
wire [7:0]	sr2_pwm_target;
wire [7:0]	sr3_pwm_target;
wire [7:0]	sd0_pwm_target;
wire [7:0]	sd1_pwm_target;
wire [7:0]	sd2_pwm_target;
wire [7:0]	sd3_pwm_target;
wire [7:0]	servo0_pwm_target;
wire [7:0]	servo1_pwm_target;
wire [7:0]	servo2_pwm_target;
wire [7:0]	servo3_pwm_target;
wire [11:0]	pwm_update;
wire 		    crc_error;
reg         all_pwm_done = 0;

always
    #7  spi_clock = ~spi_clock;

always
    #3  fpga_clock = ~fpga_clock;

initial begin
    $display("Starting Address Decoder test");
    
    $display("Deasserting reset");
    #10 reset_n = 1;

    $display("Asserting cs_n");
    #2 cs_n = 0;
    
    $display("-------------------------------");
    $display("--- Starting Broadcast Test ---");
    $display("-------------------------------");    
    @(negedge spi_clock);
    spi_out[15:0] = 16'h7057;
    @(posedge spi_clock);
    cs_n = 1;

    @(posedge fpga_clock);
    if( crc_error )
      $display("[ERROR] CRC error detected");
    else
      $display("[INFO] No CRC error");

    @(posedge fpga_clock);
    @(posedge fpga_clock);
    @(posedge fpga_clock);
    @(posedge fpga_clock);
    @(posedge fpga_clock);    
    if(pwm_update[11:0] == 12'hFFF)
      $display("[INFO] All pwm_update = 1");
    else
      $display("[ERROR] All pwm_update ~= 1");

    // Step 5 and 6
    @(posedge fpga_clock);
    pwm_done[11:0] = 12'hF;

    @(posedge fpga_clock);
    if(pwm_update[3:0] == 4'h0)
      $display("[INFO] pwm_update[3:0] deasserted");
    else
      $display("[ERROR] pwm_update[3:0] did not deassert");

    // Step 7 and 8
    @(posedge fpga_clock);
    pwm_done[11:0] = 12'hFF0;

    @(posedge fpga_clock);
    if(pwm_update[11:4] == 8'h0)
      $display("[INFO] pwm_update[11:4] deasserted");
    else
      $display("[ERROR] pwm_update[11:4] did not deassert");
  

    #100;
    pwm_done[11:0] = 12'h0;
    $display("-------------------------------");
    $display("---    Starting CRC Test    ---");
    $display("-------------------------------");   

    @(negedge spi_clock);
    spi_out[15:0] = 16'h4057;
    @(posedge spi_clock);
    cs_n = 1;

    @(posedge fpga_clock);
    if( ~crc_error )
      $display("[ERROR] No CRC error was detected");
    else
      $display("[INFO] CRC error was detected");


    cs_n = 0;
    #100;
    $display("-------------------------------------");
    $display("--- Starting Swerve Rotation Test ---");
    $display("-------------------------------------");   

    @(negedge spi_clock);
    spi_out[15:0] = 16'h6157;
    @(posedge spi_clock);
    cs_n = 1;

    @(posedge fpga_clock);
    if(crc_error )
      $display("[ERROR] CRC error detected");
    else
      $display("[INFO] No CRC error");

    @(posedge fpga_clock);
    @(posedge fpga_clock);
    @(posedge fpga_clock);
    @(posedge fpga_clock);
    @(posedge fpga_clock);    
    if(pwm_update[3:0] == 4'hF)
      $display("[INFO] pwm_update[3:0] asserted");
    else
      $display("[ERROR] pwm_update[3:0] did not assert");


    cs_n = 0;
    #100;
    $display("------------------------------------");
    $display("---- Starting Swerve Drive Test ----");
    $display("------------------------------------");   

    @(negedge spi_clock);
    spi_out[15:0] = 16'h5257;
    @(posedge spi_clock);
    cs_n = 1;

    @(posedge fpga_clock);
    if(crc_error )
      $display("[ERROR] CRC error detected");
    else
      $display("[INFO] No CRC error");

    @(posedge fpga_clock);
    @(posedge fpga_clock);
    @(posedge fpga_clock);
    @(posedge fpga_clock);
    @(posedge fpga_clock);    
    if(pwm_update[7:4] == 4'hF)
      $display("[INFO] pwm_update[7:4] asserted");
    else
      $display("[ERROR] pwm_update[7:4] did not assert");


    cs_n = 0;
    #100;
    $display("----------------------------------");
    $display("---- Starting Individual Test ----");
    $display("----------------------------------");   

    @(negedge spi_clock);
    spi_out[15:0] = 16'hBC57;
    @(posedge spi_clock);
    cs_n = 1;

    @(posedge fpga_clock);
    if(crc_error )
      $display("[ERROR] CRC error detected");
    else
      $display("[INFO] No CRC error");

    @(posedge fpga_clock);
    @(posedge fpga_clock);
    @(posedge fpga_clock);
    @(posedge fpga_clock);
    @(posedge fpga_clock);    
    if(pwm_update[8] == 1'h1)
      $display("[INFO] pwm_update[8] asserted");
    else
      $display("[ERROR] pwm_update[8] did not assert");

    #1000 $finish;
end
    
addr_dec ad0(	
  .reset_n			        (reset_n),
	.spi_clock            (spi_clock),
	.fpga_clock           (fpga_clock),
	.cs_n                 (cs_n),   
	.spi_out              (spi_out),
	.pwm_done             (pwm_done),
	.sr0_pwm_target       (sr0_pwm_target),
	.sr1_pwm_target       (sr1_pwm_target),
	.sr2_pwm_target       (sr2_pwm_target),
	.sr3_pwm_target       (sr3_pwm_target),
	.sd0_pwm_target       (sd0_pwm_target),
	.sd1_pwm_target       (sd1_pwm_target),
	.sd2_pwm_target       (sd2_pwm_target),
	.sd3_pwm_target       (sd3_pwm_target),
	.servo0_pwm_target    (servo0_pwm_target),
	.servo1_pwm_target    (servo1_pwm_target),
	.servo2_pwm_target    (servo2_pwm_target),
	.servo3_pwm_target    (servo3_pwm_target),
	.pwm_update           (pwm_update),
	.crc_error            (crc_error)
);

initial begin
    $dumpfile("ad.vcd");
    $dumpvars(0,test);
  end

endmodule