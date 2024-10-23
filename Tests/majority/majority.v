`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 10/16/2024 03:59:00 PM
// Design Name: 
// Module Name: majority
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


module majority_3input(
    input A,
    input B,
    input C,
    output Y
);
    wire w1, w2, w3;
    
    and G1(w1, A, B);
    and G2(w2, B, C);
    and G3(w3, A, C);
    
    or G4(Y, w1, w2, w3);

endmodule
