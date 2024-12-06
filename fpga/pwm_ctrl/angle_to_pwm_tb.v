module test();
    reg             reset_n = 0;        // Active low reset
    reg             clock = 0;          // The main clock
    reg     [11:0]  target_angle = 0;   // The angle the wheel needs to move to in degrees. This number is multiplied by 2 internally
    reg     [11:0]  current_angle = 0;  // The angle read from the motor encoder
    wire            pwm_done;       // Indicator from PWM that the pwm_ratio has been applied
    reg             angle_update = 0;   // Request to update the angle
    wire            angle_done;     // Indicator that the angle has been applied 
    wire            pwm_update;     // Request an update to the PWM ratio
    wire            pwm_enable;
    wire     [7:0]  pwm_ratio;      // The high-time of the PWM signal out of 255.
    wire            pwm_direction;   // The direction of the motor
    reg      [7:0]  timeout = 50;
    wire            pwm_signal;
    reg             done;
always
    #1  clock = ~clock;

always @(posedge clock or negedge reset_n)
    if(~reset_n)
        done <= 0;
    else
        if(angle_done)
            done <= ~done;

initial begin
    $display("Starting angle_to_pwm test");
    
    $display("Setting reset");
    #10 reset_n = 1;

    $display("---------------------------------");
    $display("--- Starting Small Angle Test ---");
    $display("---------------------------------"); 
    target_angle = 12'd100;
    current_angle = 12'd10;
    #10 angle_update = 1'b1;
    while(~done) begin
        #4000 current_angle[11:0] = current_angle[11:0] + 12'b1;
        timeout = timeout - 1;
    end
    #10000;

    /*$display("---------------------------------");
    $display("--- Starting Small Angle Test ---");
    $display("---------------------------------"); 
    target_angle = 12'd10;
    current_angle = 12'd18;
    #10 angle_update = 1'b1;
    while(done) begin
        #40 current_angle[11:0] = current_angle[11:0] - 12'b1;
        timeout = timeout - 1;
    end


    $display("----------------------------------");
    $display("--- Starting Medium Angle Test ---");
    $display("----------------------------------"); 
    target_angle = 8'd23;
    current_angle = 8'd10;
    #10 angle_update = 1'b1;
    while(done) begin
        #4000 current_angle[7:0] = current_angle[7:0] + 8'b1;
        timeout = timeout - 1;
    end

    $display("---------------------------------");
    $display("--- Starting Large Angle Test ---");
    $display("---------------------------------"); 
    target_angle = 8'd66;
    current_angle = 8'd10;
    #10 angle_update = 1'b1;
    while(~done) begin
        #4000 current_angle[7:0] = current_angle[7:0] + 8'b1;
        timeout = timeout - 1;
    end
*/
    #10000 $finish;
end

angle_to_pwm dut0(
    .reset_n        (reset_n),  	    // Active low reset
    .clock          (clock),	        // The main clock
    .target_angle   (target_angle),     // The angle the wheel needs to move to in degrees. This number is multiplied by 2 internally
    .current_angle  (current_angle),    // The angle read from the motor encoder
    .pwm_done       (pwm_done),         // Indicator from PWM that the pwm_ratio has been applied
    .angle_update   (angle_update),     // Request to update the angle
    .angle_done     (angle_done),       // Indicator that the angle has been applied 
    .pwm_enable     (pwm_enable),
    .pwm_update     (pwm_update),       // Request an update to the PWM ratio
    .pwm_ratio      (pwm_ratio),        // The high-time of the PWM signal out of 255.
    .pwm_direction  (pwm_direction)     // The direction of the motor
    );

pwm dut(
    .reset_n        (reset_n),
    .clock          (clock),
    .pwm_enable     (pwm_enable),
    .pwm_ratio      (pwm_ratio),
    .pwm_update     (pwm_update), 
    .pwm_done       (pwm_done),
    .pwm_signal     (pwm_signal)
);

initial begin
    $dumpfile("a_to_p.vcd");
    $dumpvars(0,test);
  end

endmodule