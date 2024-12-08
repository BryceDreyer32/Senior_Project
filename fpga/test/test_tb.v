module test();

reg [11:0] current_angle, target_angle, g, h;
reg [11:0] calc1, calc2, calc3, calc4;
reg [11:0] decision;
reg        direction;
wire o;

initial begin
    current_angle = 12'd10;
    target_angle = 12'd20;
    calc1 = current_angle - target_angle;
    calc2 = target_angle - current_angle;
    calc3 = 4096 - target_angle + current_angle;
    calc4 = 4096 - current_angle + target_angle;

    decision = calc1;
    direction = 1; // CCW
    if(calc2 < decision) begin
      decision = calc2;
      direction = 0; // CW      
    end
    if(calc3 < decision) begin
      decision = calc3;
      direction = 1; // CCW
    end
    if(calc4 < decision) begin
      decision = calc4;
      direction = 0; // CW
    end   
    #10;

    current_angle = 12'd20;
    target_angle = 12'd10;
    calc1 = current_angle - target_angle;
    calc2 = target_angle - current_angle;
    calc3 = 4096 - target_angle + current_angle;
    calc4 = 4096 - current_angle + target_angle;

    decision = calc1;
    direction = 1; // CCW
    if(calc2 < decision) begin
      decision = calc2;
      direction = 0; // CW      
    end
    if(calc3 < decision) begin
      decision = calc3;
      direction = 1; // CCW
    end
    if(calc4 < decision) begin
      decision = calc4;
      direction = 0; // CW
    end   
    #10;

    current_angle = 12'd4086;
    target_angle = 12'd10;
    calc1 = current_angle - target_angle;
    calc2 = target_angle - current_angle;    
    calc3 = 4096 - target_angle + current_angle;
    calc4 = 4096 - current_angle + target_angle;   

    decision = calc1;
    direction = 1; // CCW
    if(calc2 < decision) begin
      decision = calc2;
      direction = 0; // CW      
    end
    if(calc3 < decision) begin
      decision = calc3;
      direction = 1; // CCW
    end
    if(calc4 < decision) begin
      decision = calc4;
      direction = 0; // CW
    end
    #10;

    current_angle = 12'd10;
    target_angle = 12'd4086;
    calc1 = current_angle - target_angle;
    calc2 = target_angle - current_angle;    
    calc3 = 4096 - target_angle + current_angle;
    calc4 = 4096 - current_angle + target_angle;   

    decision = calc1;
    direction = 1; // CCW
    if(calc2 < decision) begin
      decision = calc2;
      direction = 0; // CW      
    end
    if(calc3 < decision) begin
      decision = calc3;
      direction = 1; // CCW
    end
    if(calc4 < decision) begin
      decision = calc4;
      direction = 0; // CW
    end
    #10;

    $finish;
end

temp temp (
  .clk (current_angle[0]), 
  .o(o)
);

initial begin
    $dumpfile("test.vcd");
    $dumpvars(0,test);
  end

endmodule