module top(
    input clock,
    output sck,
    inout sda,
    output [5:0] led
);

pwm_ctrl Test(
    .reset_n        (1'b1),        // Active low reset
    .clock          (clock),          // The main clock

    //FPGA Subsystem Interface
    .target_angle   (8'd10),   // The angle the wheel needs to move to in degrees. This number is multiplied by 2 internally
    .angle_update   (1'b1),   // Signals when an angle update is available
    .angle_done     (),     // Output sent when angle has been adjusted to target_angle

    //PWM Interface
    .pwm_done       (1'b1),       // Updated PWM ratio has been applied (1 cycle long pulse)
    .pwm_enable     (),     // Enables the PWM output
    .pwm_ratio      (),      // The high-time of the PWM signal out of 255.
    .pwm_update     (),     // Request an update to the PWM ratio

    //I2C Interface
    .sck            (sck),            // The I2C clock
    .sda            (sda)             // The I2C bi-directional data
); 


localparam WAIT_TIME = 13500000;
reg [5:0] ledCounter = 0;
reg [23:0] clockCounter = 0;

always @(posedge clock) begin
    clockCounter <= clockCounter + 1;
    if (clockCounter == WAIT_TIME) begin
        clockCounter <= 0;
        ledCounter <= ledCounter + 1;
    end
end

assign led = ledCounter[5:0];


endmodule