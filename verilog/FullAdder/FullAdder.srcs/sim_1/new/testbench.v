`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 10/16/2024 03:45:25 PM
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
    reg A, B, Cin;
    
    // Outputs
    wire Sum, Cout;
    
    // File handle and change tracker
    integer file;
    reg [2:0] last_change;
    
    // Instantiate the Unit Under Test (UUT)
    FullAdder uut (
        .a(A),
        .b(B),
        .cin(Cin),
        .sum(Sum),
        .cout(Cout)
    );
    
    initial begin
        // Open file for writing
        file = $fopen("full_adder_simulation.sim", "w");
        
        // Initialize Inputs
        A = 0;
        B = 0;
        Cin = 0;
        last_change = 3'b000;
        
        // Stimulus
        #0  A = 1'b0; B = 1'b0; Cin = 1'b0;
        #10 A = 1'b1; B = 1'b1; Cin = 1'b0;
        #10 A = 1'b1; B = 1'b1; Cin = 1'b0;
        #10 A = 1'b0; B = 1'b1; Cin = 1'b1;
        #10 A = 1'b1; B = 1'b1; Cin = 1'b1;
        
        // Close file and end simulation
        #10 $fclose(file);
        $finish;
    end
    
    // Monitor changes and write to file
    always @(A, B, Cin, Sum, Cout) begin
        if (A !== A) last_change = 3'b000;
        else if (B !== B) last_change = 3'b001;
        else if (Cin !== Cin) last_change = 3'b010;
        else if (Sum !== Sum) last_change = 3'b011;
        else last_change = 3'b100;
        
        case (last_change)
            3'b000: $fwrite(file, "%0t, A, %b\n", $time, A);
            3'b001: $fwrite(file, "%0t, B, %b\n", $time, B);
            3'b010: $fwrite(file, "%0t, Cin, %b\n", $time, Cin);
            3'b011: $fwrite(file, "%0t, Sum, %b\n", $time, Sum);
            3'b100: $fwrite(file, "%0t, Cout, %b\n", $time, Cout);
        endcase
    end
endmodule