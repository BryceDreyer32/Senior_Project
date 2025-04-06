// Copyright 2024
// Bryce's Senior Project
// Description: This is the TestBench for PWM

module test();
  reg clock = 0;
  reg reset_n = 0;
  reg pwm_enable = 0;
  reg direction = 0;
  reg [11:0] pwm_ratio = 0;
  reg pwm_update = 0;
  wire pwm_signal;

 always
    #1  clock = ~clock;

 initial begin
    $display("Starting PWM test");
    $monitor("PWM Value %b", pwm_signal);
    
    $display("Deasserting reset");
    #10 reset_n = 1;

    $display("Setting ratio to 128");    
    #2 pwm_enable =  1;
    #4 pwm_ratio[11:0] = 8'd128;
    #4 pwm_update = 1;
    #20000;

    $display("Setting ratio to 255");
    #4 pwm_ratio[11:0] = 8'd255;
    #20000;

    $display("Switch direction");
    #0 pwm_ratio[11:0] = 8'd0;
    #10 direction = 1'b1;
    
    $display("Setting ratio to 128");    
    #2 pwm_enable =  1;
    #4 pwm_ratio[11:0] = 8'd128;
    #4 pwm_update = 1;
    #20000;

    $display("Setting ratio to 255");
    #4 pwm_ratio[11:0] = 8'd255;
    #20000;

    $display("Setting pwm_update to 0 and updating pwm_ratio to 150");
    #4 pwm_update = 0;    
    #4 pwm_ratio[11:0] = 8'd150;
    #20000;

    $display("Setting pwm_enable to 0");
    #2 pwm_enable =  0;
    #20000 $finish;
end

    spark_pwm dut(
        .reset_n        (reset_n),
        .clock          (clock),
        .pwm_enable     (pwm_enable),
        .pwm_direction  (direction),
        .pwm_ratio      (pwm_ratio),
        .pwm_update     (pwm_update), 
        .pwm_signal     (pwm_signal)
    );

initial begin
    $dumpfile("pwm.vcd");
    $dumpvars(0,test);
  end

endmodule