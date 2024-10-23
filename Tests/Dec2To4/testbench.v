`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 10/16/2024 03:54:53 PM
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


module testbench();
    // Inputs
    reg [1:0] A;
    reg EN;
    
    // Outputs
    wire [3:0] Y;
    
    // File handle and change tracker
    integer file;
    reg [1:0] last_change;
    
    // Instantiate the Unit Under Test (UUT)
    Dec2To4 uut (
        .A(A),
        .EN(EN),
        .Y(Y)
    );
    
    initial begin
        // Open file for writing
        file = $fopen("decoder_2to4_simulation.sim", "w");
        
        // Initialize Inputs
        A = 2'b00;
        EN = 1'b0;
        last_change = 2'b00;
        
        // Stimulus
        #0  A = 2'b00; EN = 1'b0;
        #10 EN = 1'b1;
        #10 A = 2'b01;
        #10 A = 2'b10;
        #10 A = 2'b11;
        #10 EN = 1'b0;
        #10 A = 2'b00;
        #10 A = 2'b01;
        #10 A = 2'b10;
        #10 A = 2'b11;
        
        // Close file and end simulation
        #10 $fclose(file);
        $finish;
    end
    
    // Monitor changes and write to file
    always @(A, EN, Y) begin
        if (A !== A) last_change = 2'b00;
        else if (EN !== EN) last_change = 2'b01;
        else last_change = 2'b10;
        
        case (last_change)
            2'b00: $fwrite(file, "%0t, A, %b\n", $time, A);
            2'b01: $fwrite(file, "%0t, EN, %b\n", $time, EN);
            2'b10: $fwrite(file, "%0t, Y, %b\n", $time, Y);
        endcase
    end
endmodule