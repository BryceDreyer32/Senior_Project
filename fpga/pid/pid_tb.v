module test();
    reg             reset_n = 0;        // Active low reset
    reg             clock = 0;          // The main clock
    reg     [11:0]  target_angle = 0;   // The angle the wheel needs to move to in degrees. This number is multiplied by 2 internally
    reg     [11:0]  current_angle = 0;  // The angle read from the motor encoder
    wire            pwm_done;       // Indicator from PWM that the pwm_ratio has been applied
    reg             i2c_rd_done = 0;
    reg             angle_update = 0;   // Request to update the angle
    reg             abort_angle = 0;
    wire            angle_done;     // Indicator that the angle has been applied 
    wire            pwm_update;     // Request an update to the PWM ratio
    reg             pwm_enable;
    wire     [7:0]  pwm_ratio;      // The high-time of the PWM signal out of 255.
    wire            pwm_direction;   // The direction of the motor
    reg      [12:0]  timeout = 50;
    wire            pwm_signal;
    wire     [15:0] debug_signals;  // Debug signals
    reg             done;
    reg      [7:0]   kp = 8'h12;                // Proportional Constant: fixed point 4.4
    reg      [3:0]   ki = 4'h0;                 // Integral Constant: fixed point 0.4
    reg      [3:0]   kd = 4'h0;                 // Derivative Constant: fixed point 0.4
    wire     [63:0] profile = {8'h02, 8'h03, 8'h04, 8'h05, 8'h06, 8'h07, 8'h08, 8'h09};

always
    #1  clock = ~clock;

always @(posedge clock or negedge reset_n)
    if(~reset_n)
        done <= 0;
    else
        if(angle_done)
            done <= ~done;

initial begin
    
    $display("Setting reset");
    #10 reset_n = 1;

    $display("----------------------------------");
    $display("--- Starting startup Fail Test ---");
    $display("---          Time = %0t         ---", $time);
    $display("----------------------------------"); 
    timeout = 50;
    target_angle = 12'd100;
    current_angle = 12'd10;
    @(posedge clock) pwm_enable = 1'b1;
    @(posedge clock) angle_update = 1'b1;
    @(posedge clock) angle_update = 1'b0;
    while((timeout >= 0) & (current_angle != target_angle)) begin
        #2000 current_angle[11:0] = current_angle[11:0] + 12'b1;
        @(posedge clock) i2c_rd_done = 1;
        @(posedge clock) i2c_rd_done = 0;
        timeout = timeout - 1;
    end

    #5000;
/*
    $display("----------------------------------");
    $display("--- Starting startup Pass Test ---");
    $display("---         Time = %0t      ---", $time);
    $display("----------------------------------"); 
    timeout = 50;
//    force angle_to_pwm.PROFILE_DELAY_TARGET = 12'd5;
    target_angle = 12'd100;
    current_angle = 12'd10;
    #10 angle_update = 1'b1;
    #10 angle_update = 1'b0;
    while((timeout >= 0) & (current_angle != target_angle)) begin
        #1000 current_angle[11:0] = current_angle[11:0] + 12'b1;
        timeout = timeout - 1;
    end
    angle_update = 1'b0;

    $display("----------------------------------");
    $display("---  Starting run stall Test   ---");
    $display("---         Time = %0t     ---", $time);
    $display("----------------------------------"); 
    timeout = 50;
    target_angle = 12'd100;
    current_angle = 12'd10;
    #10 angle_update = 1'b1;
    #10 angle_update = 1'b0;
    while((timeout >= 0) & (current_angle != target_angle)) begin
        // Move for the first 5 sections, then stall out
        if(timeout > 30)
            #1500 current_angle[11:0] = current_angle[11:0] + 12'b1;
        else
            #1500;
        timeout = timeout - 1;
    end
    angle_update = 1'b0;

    $display("----------------------------------");
    $display("---    Fully stalled Test      ---");
    $display("---        Time = %0t      ---", $time);
    $display("----------------------------------"); 
    timeout = 50;
    target_angle = 12'd100;
    current_angle = 12'd10;
    #10 angle_update = 1'b1;
    #10 angle_update = 1'b0;
    #10000;


    #10000;
    timeout = 50;
    target_angle = 12'd10;
    current_angle = 12'd100;
    #10 angle_update = 1'b1;
    while((timeout >= 0) & (current_angle != target_angle)) begin
        #1000 current_angle[11:0] = current_angle[11:0] - 12'b1;
        timeout = timeout - 1;
    end
    angle_update = 1'b0;


    $display("---------------------------------");
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

pid pid(
    .reset_n        (reset_n),  	    // Active low reset
    .clock          (clock),	        // The main clock
    .target_angle   (target_angle),     // The angle the wheel needs to move to in degrees. This number is multiplied by 2 internally
    .current_angle  (current_angle),    // The angle read from the motor encoder
    .pwm_enable     (pwm_enable),       // PWM enable
    .pwm_done       (pwm_done),
    .i2c_rd_done    (i2c_rd_done),
    .angle_update   (angle_update),     // Request to update the angle
    .abort_angle    (abort_angle),      // Aborts rotating to angle
    .profile        (profile),
    .enable_stall_chk(1'b0),            // Enable the stall check
    .startup_fail   (),                 // Error: Motor stalled, unable to startup
    .kp             (kp),               // Proportional Constant: fixed point 4.4
    .ki             (ki),               // Integral Constant: fixed point 0.4
    .kd             (kd),               // Derivative Constant: fixed point 0.4
    .debug_signals  (debug_signals),    // Debug signals
    .angle_done     (angle_done),       // Indicator that the angle has been applied 
    .pwm_update     (pwm_update),       // Request an update to the PWM ratio
    .pwm_ratio      (pwm_ratio),        // The high-time of the PWM signal out of 255.
    .pwm_direction  (pwm_direction)     // The direction of the motor
    );

/*spark_pwm spark_pwm(
    .reset_n        (reset_n),
    .clock          (clock),
    .pwm_enable     (1'b1),
    .pwm_ratio      (pwm_ratio),
    .pwm_direction  (pwm_direction),
    .pwm_update     (pwm_update), 
    .pwm_done       (pwm_done),
    .pwm_signal     (pwm_signal)
); */

initial begin
    $dumpfile("pid.vcd");
    $dumpvars(0,test);
  end

endmodule