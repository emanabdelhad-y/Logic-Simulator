`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 10/16/2024 03:44:02 PM
// Design Name: 
// Module Name: FullAdder
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


module FullAdder(input a, b, cin, output sum, cout);
wire w1, w2, w3;
  xor G1(w1, a, b);
  xor G2(sum, w1, cin);
  and G3(w2, a, b);
  and G4(w3, w1, cin);
  or  G5(cout, w2, w3);
endmodule

