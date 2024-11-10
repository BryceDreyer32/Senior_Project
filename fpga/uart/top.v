module top (
    input               clock,          // The main clock          
    output  [7:0]       uart_tx         // The UART TX

);

reg temp;

uart dut(
    .reset_n        (1'b1),  	    // Active low reset
    .clock          (clock),	    // The main clock
    .tx_data        (8'h47),
    .baud_division  (8'd116),       // For baud = 115200, values of 110 to 120 seem to be stable
    .tx_start       (1'b1),
    .uart_tx        (temp)
);
assign uart_tx[7:0] = {8{temp}};
endmodule