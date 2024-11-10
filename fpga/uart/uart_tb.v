module test(
);

reg               reset_n;        // Active low reset
reg               clock;          // The main clock
reg   [7:0]       tx_data;        // This the 8-bits of data
reg   [7:0]       baud_division;  // The division ratio to achieve the desired baud rate
reg               tx_start;       // Signal to indicate that the transmission needs to start
wire              uart_tx;         // UART_TX

always
    #1  clock = ~clock;

initial begin
    $display("Starting UART test");
    
    $display("Setting reset");
    reset_n     = 0;
    clock       = 0;
    tx_start    = 0;
    #10 reset_n = 1;

    $display("---------------------------------");
    $display("--------- Sending Data ----------");
    $display("---------------------------------"); 

    tx_data         = 8'hA6;
    baud_division   = 8'd5;

    #100 tx_start   = 1;

    #5000 $finish;
end

uart dut(
    .reset_n        (reset_n),  	    // Active low reset
    .clock          (clock),	        // The main clock
    .tx_data        (tx_data),
    .baud_division  (baud_division),
    .tx_start       (tx_start),
    .uart_tx        (uart_tx)
);

initial begin
    $dumpfile("uart.vcd");
    $dumpvars(0,test);
end

endmodule