`timescale 1ns / 1ps

module hadd_tb;
    // Inputs
    reg x;
    reg y;
    
    // Outputs
    wire s;
    wire c;
    
    // File handle and change tracker
    integer file;
    reg [1:0] last_change;
    
    // Instantiate the half adder
    hadd uut (
        .x(x),
        .y(y),
        .s(s),
        .c(c)
    );
    
    initial begin
        // Open file for writing
        file = $fopen("half_adder_simulation.sim", "w");
        
        // Initialize Inputs
        x = 0;
        y = 0;
        last_change = 2'b00;
        
        // Test vector generation
        #0  x = 0; y = 0;
        #10 x = 0; y = 1;
        #10 x = 1; y = 0;
        #10 x = 1; y = 1;
        
        // Close file and end simulation
        #10 $fclose(file);
        $finish;
    end
    
    // Monitor changes and write to file
    always @(x, y, s, c) begin
        if (x !== x) last_change = 2'b00;
        else if (y !== y) last_change = 2'b01;
        else if (s !== s) last_change = 2'b10;
        else last_change = 2'b11;
        
        case (last_change)
            2'b00: $fwrite(file, "%0t, x, %b\n", $time, x);
            2'b01: $fwrite(file, "%0t, y, %b\n", $time, y);
            2'b10: $fwrite(file, "%0t, s, %b\n", $time, s);
            2'b11: $fwrite(file, "%0t, c, %b\n", $time, c);
        endcase
    end
    
    // Monitor output values (optional, for console display)
    initial begin
        $monitor("Time: %0dns | x: %b | y: %b | Sum: %b | Carry: %b", $time, x, y, s, c);
    end
endmodule