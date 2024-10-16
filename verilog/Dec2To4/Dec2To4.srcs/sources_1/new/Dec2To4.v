`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 10/16/2024 03:53:33 PM
// Design Name: 
// Module Name: Dec2To4
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


module Dec2To4(
 input [1:0] A,
    input EN,
    output [3:0] Y
    );
    
    wire [1:0] An;
    
    not G1(An[0], A[0]);
    not G2(An[1], A[1]);
    
    and G3(Y[0], An[1], An[0], EN);
    and G4(Y[1], An[1], A[0], EN);
    and G5(Y[2], A[1], An[0], EN);
    and G6(Y[3], A[1], A[0], EN);
    
endmodule
