// Copyright 2024
// Bryce's Senior Project
// Description: This module calculates the delta between two positions over multiple clock cycles

module test (
);

    reg         reset_n = 0;        // Active low reset
    reg         clock = 0;          // The main clock
    reg         enable_calc = 0;    // Enable this module to calculate (otherwise output is just 0)
    reg [11:0]  target_angle = 0;   // The angle the wheel needs to move on the 4096 points/rotation scale
    reg [11:0]  current_angle = 0;  // The angle read from the motor encoder
    wire        dir_shortest;       // The direction to travel for the shortest path from current -> target
    wire[11:0]  delta_angle;        // The shortest distance from current -> target
    wire        calc_updated;       // Pulse indicating that the calculation has been updated


always
    #1  clock = ~clock;

initial begin
    $display("Starting angle_to_pwm test");
    #5 reset_n = 1;

    $display("------------------------------------------------");
    $display("--- Starting Simple Clockwise calculation    ---");
    $display("------------------------------------------------"); 
    target_angle    = 1024;
    current_angle   = 100;

    #5 enable_calc  = 1;

    while(calc_updated == 1'b0)
        #2;
    enable_calc  = 0;

    $display("------------------------------------------------");
    $display("--- Starting Simple CCW calculation          ---");
    $display("------------------------------------------------"); 
    target_angle    = 100;
    current_angle   = 1024;

    #5 enable_calc  = 1;

    while(calc_updated == 1'b0)
        #2;
    enable_calc  = 0;

    $display("------------------------------------------------");
    $display("--- CCW through origin calculation           ---");
    $display("------------------------------------------------"); 
    target_angle    = 200;
    current_angle   = 3600;

    #5 enable_calc  = 1;

    while(calc_updated == 1'b0)
        #2;
    enable_calc  = 0;

    $display("------------------------------------------------");
    $display("--- Clockwise through origin calculation     ---");
    $display("------------------------------------------------"); 
    target_angle    = 3600;
    current_angle   = 200;

    #5 enable_calc  = 1;

    while(calc_updated == 1'b0)
        #2;
    enable_calc  = 0;


    #10;
    $finish;

end

calculate_delta dut (
    .reset_n        (reset_n),      
    .clock          (clock),        
    .enable_calc    (enable_calc),  
    .target_angle   (target_angle), 
    .current_angle  (current_angle),
    .dir_shortest   (dir_shortest), 
    .delta_angle    (delta_angle),  
    .calc_updated   (calc_updated)
);

initial begin
    $dumpfile("calculate_delta_tb.vcd");
    $dumpvars(0, test);
end

endmodule