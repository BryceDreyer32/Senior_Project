module test();
    reg             reset_n = 0;        // Active low reset
    reg             clock = 0;          // The main clock
    reg     [7:0]   target_angle = 0;   // The angle the wheel needs to move to in degrees. This number is multiplied by 2 internally
    reg     [7:0]   current_angle = 0;  // The angle read from the motor encoder
    reg             pwm_done;       // Indicator from PWM that the pwm_ratio has been applied
    reg             angle_update = 0;   // Request to update the angle
    wire            angle_done;     // Indicator that the angle has been applied 
    wire            pwm_update;     // Request an update to the PWM ratio
    wire            pwm_enable;
    wire     [7:0]  pwm_ratio;      // The high-time of the PWM signal out of 255.
    reg      [7:0]  timeout = 50;
    reg             done;
    wire            sda;
    wire            sck;
always
    #1  clock = ~clock;

always @(posedge clock or negedge reset_n)
    if(~reset_n)
        done <= 0;
    else
        if(angle_done)
            done <= ~done;

initial begin
    $display("Starting pwm_ctrl test");
    
    $display("Setting reset");
    pwm_done    = 1'd0;
    #10 reset_n = 1;


    $display("----------------------------------");
    $display("--- Starting Medium Angle Test ---");
    $display("----------------------------------"); 
    target_angle =   8'd23;
    current_angle =  8'd10;
    #10 angle_update = 1'b1;
    //while(~angle_done) begin
    //    #4000 current_angle[7:0] = current_angle[7:0] + 8'b1;
    //    timeout = timeout - 1;
    //end


    #30000 $finish;
end

pwm_ctrl dut(
    .reset_n        (reset_n),          // Active low reset
    .clock          (clock),            // The main clock

    //FPGA Subsystem Interface
    .target_angle   (target_angle),     // The angle the wheel needs to move to in degrees. This number is multiplied by 2 internally
    .angle_update   (angle_update),     // Signals when an angle update is available
    .angle_done     (angle_done),       // Output sent when angle has been adjusted to target_angle

    //PWM Interface
    .pwm_done       (pwm_done),          // Updated PWM ratio has been applied (1 cycle long pulse)
    .pwm_enable     (pwm_enable),        // Enables the PWM output
    .pwm_ratio      (pwm_ratio),         // The high-time of the PWM signal out of 255.
    .pwm_update     (pwm_update),        // Request an update to the PWM ratio

    //I2C Interface
    .sck            (sck),               // The I2C clock
    .sda            (sda)                // The I2C bi-directional data
);

initial begin
    $dumpfile("pwm_ctrl.vcd");
    $dumpvars(0,test);
  end

endmodule