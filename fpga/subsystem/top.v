module top(
    // Clock and Reset
    input           reset_n,        // Active low reset
    input           clock,          // The main clock
    
    // SPI Interface
    input           spi_clock,      // The SPI clock
    input           cs_n,           // Active low chip select
    input           mosi,           // Master out slave in (SPI mode 0)

    // Swerve Rotation Motors
    output  [3:0]   sr_pwm,         // The swerve rotation PWM wave
    output  [3:0]   scl,            // The I2C clock
    inout   [3:0]   sda,            // The I2C bi-directional data

    // Swerve Drive Motors
    output  [3:0]   sd_pwm,         // The swerve drive PWM wave

    // Arm Servos
    output  [3:0]   servo_pwm       //The arm servo PWM wave
);

genvar i;

wire    [15:0]      spi_out;
wire    [15:0]      sr_pwm_target[3:0];
wire    [11:0]      pwm_update;
wire    [3:0]       sr_pwm_enable;
wire    [7:0]       sr_pwm_ratio[3:0];
wire    [7:0]       sd_pwm_ratio[3:0];
wire    [7:0]       servo_pwm_ratio[3:0];
wire    [3:0]       sr_pwm_update;

////////////////////////////////////////////////////////////////
// SPI Controller
////////////////////////////////////////////////////////////////
spi spi(
    .reset_n                (reset_n),
    .clock                  (clock),
    .cs_n                   (cs_n),
    .mosi                   (mosi),
    .shadow_reg             (spi_out[15:0])
);

////////////////////////////////////////////////////////////////
// Address Decoder
////////////////////////////////////////////////////////////////
addr_dec ad0(	
    .reset_n			    (reset_n),
	.spi_clock              (spi_clock),
	.fpga_clock             (clock),
	.cs_n                   (cs_n),   
	.spi_out                (spi_out[15:0]),
	.pwm_done               (pwm_done),
	.sr0_pwm_target         (sr_pwm_target[0]),
	.sr1_pwm_target         (sr_pwm_target[1]),
	.sr2_pwm_target         (sr_pwm_target[2]),
	.sr3_pwm_target         (sr_pwm_target[3]),
	.sd0_pwm_target         (sd_pwm_ratio[0]),
	.sd1_pwm_target         (sd_pwm_ratio[1]),
	.sd2_pwm_target         (sd_pwm_ratio[2]),
	.sd3_pwm_target         (sd_pwm_ratio[3]),
	.servo0_pwm_target      (servo_pwm_ratio[0]),
	.servo1_pwm_target      (servo_pwm_ratio[1]),
	.servo2_pwm_target      (servo_pwm_ratio[2]),
	.servo3_pwm_target      (servo_pwm_ratio[3]),
	.pwm_update             (pwm_update[11:0]),
	.crc_error              () 
);

////////////////////////////////////////////////////////////////
// Swerve Rotation Motor
////////////////////////////////////////////////////////////////
pwm_ctrl pwm_ctrl0(
    .reset_n                (reset_n),          // Active low reset
    .clock                  (clock),            // The main clock

    //FPGA Subsystem Interface
    .target_angle           (sr_pwm_target[0]), // The angle the wheel needs to move to in degrees. This number is multiplied by 2 internally
    .angle_update           (pwm_update[0]),    // Signals when an angle update is available
    .angle_done             (pwm_done[0]),      // Output sent when angle has been adjusted to target_angle

    //PWM Interface
    .pwm_done               (),                 // Updated PWM ratio has been applied (1 cycle long pulse)
    .pwm_enable             (sr_pwm_enable[0]), // Enables the PWM output
    .pwm_ratio              (sr_pwm_ratio[0]),  // The high-time of the PWM signal out of 255.
    .pwm_update             (sr_pwm_update[0]), // Request an update to the PWM ratio

    //I2C Interface
    .sck                    (sck[0]),           // The I2C clock
    .sda                    (sda[0])            // The I2C bi-directional data
);

pwm sr_pwm0(
    .reset_n                (reset_n),
    .clock                  (clock),
    .pwm_enable             (sr_pwm_enable[0]),
    .pwm_ratio              (sr_pwm_ratio[0]),
    .pwm_update             (sr_pwm_update[0]), 
    .pwm_signal             (sr_pwm[0])
);

pwm_ctrl pwm_ctrl1(
    .reset_n                (reset_n),          // Active low reset
    .clock                  (clock),            // The main clock

    //FPGA Subsystem Interface
    .target_angle           (sr_pwm_target[1]), // The angle the wheel needs to move to in degrees. This number is multiplied by 2 internally
    .angle_update           (pwm_update[1]),    // Signals when an angle update is available
    .angle_done             (pwm_done[1]),      // Output sent when angle has been adjusted to target_angle

    //PWM Interface
    .pwm_done               (),                 // Updated PWM ratio has been applied (1 cycle long pulse)
    .pwm_enable             (sr_pwm_enable[1]), // Enables the PWM output
    .pwm_ratio              (sr_pwm_ratio[1]),  // The high-time of the PWM signal out of 255.
    .pwm_update             (sr_pwm_update[1]), // Request an update to the PWM ratio

    //I2C Interface
    .sck                    (sck[1]),           // The I2C clock
    .sda                    (sda[1])            // The I2C bi-directional data
);

pwm sr_pwm1(
    .reset_n                (reset_n),
    .clock                  (clock),
    .pwm_enable             (sr_pwm_enable[1]),
    .pwm_ratio              (sr_pwm_ratio[1]),
    .pwm_update             (sr_pwm_update[1]), 
    .pwm_signal             (sr_pwm[1])
);

pwm_ctrl pwm_ctrl2(
    .reset_n                (reset_n),          // Active low reset
    .clock                  (clock),            // The main clock

    //FPGA Subsystem Interface
    .target_angle           (sr_pwm_target[2]), // The angle the wheel needs to move to in degrees. This number is multiplied by 2 internally
    .angle_update           (pwm_update[2]),    // Signals when an angle update is available
    .angle_done             (pwm_done[2]),      // Output sent when angle has been adjusted to target_angle

    //PWM Interface
    .pwm_done               (),                 // Updated PWM ratio has been applied (1 cycle long pulse)
    .pwm_enable             (sr_pwm_enable[2]), // Enables the PWM output
    .pwm_ratio              (sr_pwm_ratio[2]),  // The high-time of the PWM signal out of 255.
    .pwm_update             (sr_pwm_update[2]), // Request an update to the PWM ratio

    //I2C Interface
    .sck                    (sck[2]),           // The I2C clock
    .sda                    (sda[2])            // The I2C bi-directional data
);

pwm sr_pwm2(
    .reset_n                (reset_n),
    .clock                  (clock),
    .pwm_enable             (sr_pwm_enable[2]),
    .pwm_ratio              (sr_pwm_ratio[2]),
    .pwm_update             (sr_pwm_update[2]), 
    .pwm_signal             (sr_pwm[2])
);

pwm_ctrl pwm_ctrl3(
    .reset_n                (reset_n),          // Active low reset
    .clock                  (clock),            // The main clock

    //FPGA Subsystem Interface
    .target_angle           (sr_pwm_target[3]), // The angle the wheel needs to move to in degrees. This number is multiplied by 2 internally
    .angle_update           (pwm_update[3]),    // Signals when an angle update is available
    .angle_done             (pwm_done[3]),      // Output sent when angle has been adjusted to target_angle

    //PWM Interface
    .pwm_done               (),                 // Updated PWM ratio has been applied (1 cycle long pulse)
    .pwm_enable             (sr_pwm_enable[3]), // Enables the PWM output
    .pwm_ratio              (sr_pwm_ratio[3]),  // The high-time of the PWM signal out of 255.
    .pwm_update             (sr_pwm_update[3]), // Request an update to the PWM ratio

    //I2C Interface
    .sck                    (sck[3]),           // The I2C clock
    .sda                    (sda[3])            // The I2C bi-directional data
);

pwm sr_pwm3(
    .reset_n                (reset_n),
    .clock                  (clock),
    .pwm_enable             (sr_pwm_enable[3]),
    .pwm_ratio              (sr_pwm_ratio[3]),
    .pwm_update             (sr_pwm_update[3]), 
    .pwm_signal             (sr_pwm[3])
);


////////////////////////////////////////////////////////////////
// Swerve Drive Motor
////////////////////////////////////////////////////////////////
pwm sd_pwm0(
    .reset_n                (reset_n),
    .clock                  (clock),
    .pwm_enable             (1'b1),
    .pwm_ratio              (sd_pwm_ratio[0]),
    .pwm_update             (1'b1), 
    .pwm_signal             (sd_pwm[0])
);

pwm sd_pwm1(
    .reset_n                (reset_n),
    .clock                  (clock),
    .pwm_enable             (1'b1),
    .pwm_ratio              (sd_pwm_ratio[1]),
    .pwm_update             (1'b1), 
    .pwm_signal             (sd_pwm[1])
);

pwm sd_pwm2(
    .reset_n                (reset_n),
    .clock                  (clock),
    .pwm_enable             (1'b1),
    .pwm_ratio              (sd_pwm_ratio[2]),
    .pwm_update             (1'b1), 
    .pwm_signal             (sd_pwm[2])
);

pwm sd_pwm3(
    .reset_n                (reset_n),
    .clock                  (clock),
    .pwm_enable             (1'b1),
    .pwm_ratio              (sd_pwm_ratio[3]),
    .pwm_update             (1'b1), 
    .pwm_signal             (sd_pwm[3])
);


////////////////////////////////////////////////////////////////
// Servos
////////////////////////////////////////////////////////////////
pwm servo_pwm0(
    .reset_n                (reset_n),
    .clock                  (clock),
    .pwm_enable             (1'b1),
    .pwm_ratio              (servo_pwm_ratio[0]),
    .pwm_update             (1'b1), 
    .pwm_signal             (servo_pwm[0])
);

pwm servo_pwm1(
    .reset_n                (reset_n),
    .clock                  (clock),
    .pwm_enable             (1'b1),
    .pwm_ratio              (servo_pwm_ratio[1]),
    .pwm_update             (1'b1), 
    .pwm_signal             (servo_pwm[1])
);

pwm servo_pwm2(
    .reset_n                (reset_n),
    .clock                  (clock),
    .pwm_enable             (1'b1),
    .pwm_ratio              (servo_pwm_ratio[2]),
    .pwm_update             (1'b1), 
    .pwm_signal             (servo_pwm[2])
);

pwm servo_pwm3(
    .reset_n                (reset_n),
    .clock                  (clock),
    .pwm_enable             (1'b1),
    .pwm_ratio              (servo_pwm_ratio[3]),
    .pwm_update             (1'b1), 
    .pwm_signal             (servo_pwm[3])
);

endmodule