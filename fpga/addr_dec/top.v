module top (); 

    addr_dec ad0();

initial begin
    $dumpfile("ad.vcd");
    $dumpvars(0,test);
  end

endmodule