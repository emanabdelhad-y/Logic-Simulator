`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 10/16/2024 03:34:00 PM
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

module tb_MUX4to1();
    // Inputs
    reg [3:0] D;
    reg [1:0] S;
    
    // Outputs
    wire Y;
    
    // File handle and change tracker
    integer file;
    reg [1:0] last_change;
    
    // Instantiate the Unit Under Test (UUT)
    MUX4to1 uut (
        .D(D),
        .S(S),
        .Y(Y)
    );
    
    initial begin
        // Open file for writing
        file = $fopen("mux4to1_simulation.sim", "w");
        
        // Initialize Inputs
        D = 4'b0000;
        S = 2'b00;
        last_change = 2'b00;
        
        // Stimulus
        #0  D = 4'b1010; S = 2'b00;
        #10 S = 2'b01;
        #10 S = 2'b10;
        #10 S = 2'b11;
        #10 D = 4'b0110; S = 2'b00;
        #10 S = 2'b01;
        #10 S = 2'b10;
        #10 S = 2'b11;
        
        // Close file and end simulation
        #10 $fclose(file);
        $finish;
    end
    
    // Monitor changes and write to file
    always @(D, S, Y) begin
        if (D !== D) last_change = 2'b00;
        else if (S !== S) last_change = 2'b01;
        else last_change = 2'b10;
        
        case (last_change)
            2'b00: $fwrite(file, "%0t, D, %b\n", $time, D);
            2'b01: $fwrite(file, "%0t, S, %b\n", $time, S);
            2'b10: $fwrite(file, "%0t, Y, %b\n", $time, Y);
        endcase
    end
    
    // Optional: Display output in console
    initial begin
        $monitor("Time: %0t | D: %b | S: %b | Y: %b", $time, D, S, Y);
    end
endmodule