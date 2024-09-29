// Copyright 2024
// Bryce's Senior Project
// Description: This is the TestBench for PWM

module test();
  reg clock = 0;
  reg reset_n = 0;
  reg pwm_enable = 0;
  reg [7:0] pwm_ratio = 0;
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
    #4 pwm_ratio[7:0] = 8'd128;
    #4 pwm_update = 1;
    #5000;

    $display("Setting ratio to 200");
    #4 pwm_ratio[7:0] = 8'd200;
    #5000;

    $display("Setting pwm_update to 0 and updating pwm_ratio to 150");
    #4 pwm_update = 0;    
    #4 pwm_ratio[7:0] = 8'd150;
    #5000;

    $display("Setting pwm_enable to 0");
    #2 pwm_enable =  0;
    #1000 $finish;
end

    pwm dut(
        .reset_n        (reset_n),
        .clock          (clock),
        .pwm_enable     (pwm_enable),
        .pwm_ratio      (pwm_ratio),
        .pwm_update     (pwm_update), 
        .pwm_signal     (pwm_signal)
    );

initial begin
    $dumpfile("pwm.vcd");
    $dumpvars(0,test);
  end

endmodule