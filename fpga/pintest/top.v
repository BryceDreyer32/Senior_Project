// Copyright 2024
// Bryce's Senior Project
// Description: This is the RTL for the FPGA Subsystem
module top(
    // Clock and Reset
    //input           reset_n,        // Active low reset
    input           clock,          // The main clock
    output out0,
    output out1,
    output out2,
    output out3,
    output out4,
    output out5,
    output out6,
    output out7,
    output out8,
    output out9,
    output out10,
    output out11,
    output out12,
    output out13,
    output out14,
    output out15,
    output out16,
    output out17,
    output out18,
    output out19,
    output out20,
    output out21,
    output out22,
    output out23,
    output out24,
    output out25,
    output out26,
    output out27,
    output out28,
    output out29,
    output out30,
    output out31,
    output out32,
    output out33,
    output out34,
    output out35,
    output out36,
    output out37,
    output out38,
    output out39,
    output out40
    );

    reg[3:0] counter;

    always @(posedge clock) begin
        counter <= counter + 1;
    end

    assign out0= counter[3];
    assign out1= counter[3];
    assign out2= counter[3];
    assign out3= counter[3];
    assign out4= counter[3];
    assign out5= counter[3];
    assign out6= counter[3];
    assign out7= counter[3];
    assign out8= counter[3];
    assign out9= counter[3];
    assign out10= counter[3];
    assign out11= counter[3];
    assign out12= counter[3];
    assign out13= counter[3];
    assign out14= counter[3];
    assign out15= counter[3];
    assign out16= counter[3];
    assign out17= counter[3];
    assign out18= counter[3];
    assign out19= counter[3];
    assign out20= counter[3];
    assign out21= counter[3];
    assign out22= counter[3];
    assign out23= counter[3];
    assign out24= counter[3];
    assign out25= counter[3];
    assign out26= counter[3];
    assign out27= counter[3];
    assign out28= counter[3];
    assign out29= counter[3];
    assign out30= counter[3];
    assign out31= counter[3];
    assign out32= counter[3];
    assign out33= counter[3];
    assign out34= counter[3];
    assign out35= counter[3];
    assign out36= counter[3];
    assign out37= counter[3];
    assign out38= counter[3];
    assign out39= counter[3];
    assign out40= counter[3];
    
endmodule
