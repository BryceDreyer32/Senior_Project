module test();
    reg       reset_n   = 0;            // Active low reset
    reg       clock     = 0;              // The main clock
    reg       pwm_enable= 0;         // Enables the PWM output
//    input [7:0] min_pwm_ratio,      // The minimum ratio
//    input [7:0] max_pwm_ratio,      // The maximum ratio
    reg [7:0] start_pwm_ratio;    // The initial pwm ratio
    reg [7:0] target_pwm_ratio;   // The target pwm ratio
    wire      pwm_signal;         // The output PWM wave

always
    #1  clock = ~clock;

initial begin
    #10 reset_n = 1'b1;

    start_pwm_ratio = 8'd20;
    target_pwm_ratio = 8'd0;
    pwm_enable = 1'b1;
    #20000;

    target_pwm_ratio = 8'd50;
    #600000;

    target_pwm_ratio = 8'd10;
    #600000;

    $finish;
end

servo_ctrl dut
(
    .reset_n            (reset_n),
    .clock              (clock),
    .pwm_enable         (pwm_enable),
    .start_pwm_ratio    (start_pwm_ratio),
    .target_pwm_ratio   (target_pwm_ratio),
    .pwm_signal         (pwm_signal)
);

initial begin
    $dumpfile("servo_ctrl.vcd");
    $dumpvars(0,test);
  end

endmodule