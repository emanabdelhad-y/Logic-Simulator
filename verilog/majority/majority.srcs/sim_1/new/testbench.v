`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 10/16/2024 03:59:14 PM
// Design Name: 
// Module Name: testbench
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module tb_majority_3input();
    reg A, B, C;
    wire Y;
    
    integer file;
    reg [1:0] last_change;
    
    majority_3input uut (
        .A(A),
        .B(B),
        .C(C),
        .Y(Y)
    );
    
    initial begin
        file = $fopen("majority_3input_simulation.sim", "w");
        A = 0; B = 0; C = 0;
        last_change = 2'b00;
        #10 A = 0; B = 0; C = 1;
        #10 A = 0; B = 1; C = 0;
        #10 A = 0; B = 1; C = 1;
        #10 A = 1; B = 0; C = 0;
        #10 A = 1; B = 0; C = 1;
        #10 A = 1; B = 1; C = 0;
        #10 A = 1; B = 1; C = 1;
        #10 $fclose(file);
        $finish;
    end
    
    always @(A, B, C, Y) begin
        if (A !== A) last_change = 2'b00;
        else if (B !== B) last_change = 2'b01;
        else if (C !== C) last_change = 2'b10;
        else last_change = 2'b11;
        
        case (last_change)
            2'b00: $fwrite(file, "%0t, A, %b\n", $time, A);
            2'b01: $fwrite(file, "%0t, B, %b\n", $time, B);
            2'b10: $fwrite(file, "%0t, C, %b\n", $time, C);
            2'b11: $fwrite(file, "%0t, Y, %b\n", $time, Y);
        endcase
    end
endmodule