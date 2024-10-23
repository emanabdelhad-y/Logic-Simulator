`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 10/16/2024 03:31:30 PM
// Design Name: 
// Module Name: MUX4To1
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

module MUX4to1(input [3:0] D, input [1:0] S, output Y);
  wire w1, w2, w3, w4, s0n, s1n;
  not G1(s0n, S[0]);
  not G2(s1n, S[1]);
  and G3(w1, D[0], s0n, s1n);
  and G4(w2, D[1], S[0], s1n);
  and G5(w3, D[2], s0n, S[1]);
  and G6(w4, D[3], S[0], S[1]);
  or  G7(Y, w1, w2, w3, w4);
endmodule