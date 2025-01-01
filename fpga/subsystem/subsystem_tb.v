module test();
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
    force dut.rf.servo_control      = 8'b0;
    force dut.rf.servo_position0    = 8'b0;
    force dut.reset_cntr            = 3'b0;
    #100;
    release dut.reset_cntr;
    force dut.rf.servo_control      = 8'hF;
    #10000000;

    force dut.rf.servo_position0    = 8'hF;
    #10000;

    $finish;
end

top dut(
    .clock      (clock)
);

initial begin
    $dumpfile("subsystem.vcd");
    $dumpvars(0,test);
end


endmodule