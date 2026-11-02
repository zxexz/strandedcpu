`timescale 1ns/1ps
module integration_tb;
    localparam DIVISOR = 104;
    reg clk = 0;
    always #5 clk = !clk;
    reg rx = 1;
    wire tx;
    wire [4:0] led;
    top dut(.clk(clk), .rx(rx), .tx(tx), .led(led));

    task send_byte(input [7:0] value);
        integer bit_number;
        begin
            @(negedge clk); rx = 0;
            repeat (DIVISOR) @(negedge clk);
            for (bit_number = 0; bit_number < 8; bit_number = bit_number + 1) begin
                rx = value[bit_number];
                repeat (DIVISOR) @(negedge clk);
            end
            rx = 1;
            repeat (DIVISOR) @(negedge clk);
        end
    endtask

    task expect_byte(input [7:0] expected);
        reg [7:0] value;
        integer bit_number;
        begin
            @(negedge tx);
            repeat (DIVISOR / 2) @(posedge clk);
            #1;
            if (tx !== 0) $fatal(1, "UART start bit");
            for (bit_number = 0; bit_number < 8; bit_number = bit_number + 1) begin
                repeat (DIVISOR) @(posedge clk);
                #1; value[bit_number] = tx;
            end
            repeat (DIVISOR) @(posedge clk);
            #1;
            if (tx !== 1 || value !== expected)
                $fatal(1, "Integration response got %02x expected %02x", value, expected);
        end
    endtask

    task instruction(input [7:0] op, input [7:0] b, input [7:0] c);
        begin send_byte(op); send_byte(b); send_byte(c); end
    endtask

    task load_memory(input [7:0] length);
        integer word_number;
        begin
            send_byte(8'h4c);
            send_byte(3); send_byte(5);
            for (word_number = 2; word_number < 32; word_number = word_number + 1)
                send_byte(word_number);
            send_byte(length);
        end
    endtask

    task result(input [7:0] fault, input [15:0] steps, input [7:0] pc, input [7:0] first_word);
        integer word_number;
        begin
            expect_byte(8'h44); expect_byte(fault);
            expect_byte(steps[7:0]); expect_byte(steps[15:8]);
            expect_byte(pc); expect_byte(0);
            expect_byte(first_word); expect_byte(5);
            for (word_number = 2; word_number < 32; word_number = word_number + 1)
                expect_byte(word_number);
        end
    endtask

    task run_normal(input [7:0] first_word);
        begin
            send_byte(8'h52); send_byte(100);
            fork send_byte(0); result(0, 4, 0, first_word); join
            repeat (DIVISOR) @(negedge clk);
        end
    endtask

    initial begin
        repeat (40) @(negedge clk);
        load_memory(4);
        instruction(2, 32, 33); // ADD 3+5
        instruction(8, 32, 0); // SHL ->16
        instruction(6, 32, 0); // INC ->17
        fork instruction(0, 0, 0); expect_byte(8'h4b); join
        repeat (DIVISOR) @(negedge clk);
        run_normal(17);
        run_normal(45); // Second run uses the first run's memory and clears done.

        load_memory(1);
        fork instruction(22, 32, 33); expect_byte(8'h4b); join
        repeat (DIVISOR) @(negedge clk);
        send_byte(8'h52); send_byte(100);
        fork send_byte(0); result(1, 0, 64, 3); join
        repeat (DIVISOR) @(negedge clk);

        load_memory(2);
        instruction(6, 32, 0);
        fork instruction(3, 0, 64); expect_byte(8'h4b); join
        repeat (DIVISOR) @(negedge clk);
        send_byte(8'h52); send_byte(3);
        fork send_byte(0); result(3, 3, 65, 5); join
        $display("PASS integration: UART loader, real ISA execution, rerun, unsupported opcode, step limit");
        $finish;
    end
    initial begin #20000000; $fatal(1, "Integration test timeout"); end
endmodule
