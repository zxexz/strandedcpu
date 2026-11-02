`timescale 1ns/1ps
// A synchronous-memory stub isolates UART/protocol timing from ISA execution.
module isa_core(
    input wire clk, reset, start,
    input wire [15:0] step_limit,
    input wire [7:0] program_length,
    input wire host_write,
    input wire [4:0] host_addr,
    input wire [7:0] host_wdata,
    output reg [7:0] host_rdata,
    input wire code_write,
    input wire [8:0] code_addr,
    input wire [7:0] code_wdata,
    output reg done, busy,
    output reg [7:0] fault,
    output reg [15:0] steps,
    output reg [7:0] pc,
    output reg carry
);
    reg [7:0] memory [0:31];
    reg [7:0] code [0:383];
    reg [2:0] ticks;
    integer memory_writes = 0, code_writes = 0, starts = 0;
    always @(posedge clk) begin
        host_rdata <= memory[host_addr];
        if (host_write) begin
            memory[host_addr] <= host_wdata;
            memory_writes <= memory_writes + 1;
        end
        if (code_write) begin
            code[code_addr] <= code_wdata;
            code_writes <= code_writes + 1;
        end
        if (reset) begin
            done <= 0; busy <= 0; fault <= 0; steps <= 0; pc <= 0; carry <= 0;
            ticks <= 0;
        end else if (start) begin
            done <= 0; busy <= 1; fault <= 0; ticks <= 4;
            starts <= starts + 1;
            if (program_length != 128 || step_limit != 16'h1234)
                $fatal(1, "Core received wrong program length/step limit");
        end else if (busy) begin
            if (ticks != 0) ticks <= ticks - 1'b1;
            else begin
                memory[0] <= memory[0] + memory[1];
                done <= 1; busy <= 0; fault <= 3; steps <= step_limit; pc <= 8'h45; carry <= 1;
            end
        end
    end
endmodule

module top_tb;
    localparam DIVISOR = 104;
    reg clk = 0;
    always #5 clk = !clk;
    reg rx = 1;
    wire tx;
    wire [4:0] led;
    integer i;
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
            if (tx !== 0) $fatal(1, "UART response start bit");
            for (bit_number = 0; bit_number < 8; bit_number = bit_number + 1) begin
                repeat (DIVISOR) @(posedge clk);
                #1; value[bit_number] = tx;
            end
            repeat (DIVISOR) @(posedge clk);
            #1;
            if (tx !== 1 || value !== expected)
                $fatal(1, "Response got %02x expected %02x", value, expected);
        end
    endtask

    task ping;
        begin
            fork
                send_byte(8'h50);
                begin
                    expect_byte(8'h53); expect_byte(8'h43);
                    expect_byte(8'h38); expect_byte(8'h01);
                end
            join
            repeat (DIVISOR) @(negedge clk);
        end
    endtask

    task load_program(input [7:0] length);
        integer byte_number;
        begin
            send_byte(8'h4c);
            for (byte_number = 0; byte_number < 32; byte_number = byte_number + 1)
                send_byte(8'h80 + byte_number);
            if (length == 0 || length > 128) begin
                fork send_byte(length); expect_byte(8'h45); join
            end else begin
                send_byte(length);
                for (byte_number = 0; byte_number < 3 * length - 1; byte_number = byte_number + 1)
                    send_byte((byte_number * 7) ^ 8'ha5);
                fork
                    send_byte(((3 * length - 1) * 7) ^ 8'ha5);
                    expect_byte(8'h4b);
                join
            end
            repeat (DIVISOR) @(negedge clk);
        end
    endtask

    task run_program(input [7:0] first_memory);
        integer byte_number;
        begin
            send_byte(8'h52); send_byte(8'h34);
            fork
                send_byte(8'h12);
                begin
                    expect_byte(8'h44); expect_byte(8'h03);
                    expect_byte(8'h34); expect_byte(8'h12);
                    expect_byte(8'h45); expect_byte(8'h01);
                    expect_byte(first_memory);
                    for (byte_number = 1; byte_number < 32; byte_number = byte_number + 1)
                        expect_byte(8'h80 + byte_number);
                end
            join
            repeat (DIVISOR) @(negedge clk);
        end
    endtask

    initial begin
        repeat (40) @(negedge clk);
        if (!tx || !led[4]) $fatal(1, "Boot did not enter quiet ready state");
        ping();
        load_program(128);
        if (dut.core.memory_writes != 32 || dut.core.code_writes != 384)
            $fatal(1, "Loader write pulse/count mismatch");
        for (i = 0; i < 32; i = i + 1)
            if (dut.core.memory[i] !== (8'h80 + i)) $fatal(1, "Loaded memory[%0d]", i);
        for (i = 0; i < 384; i = i + 1)
            if (dut.core.code[i] !== (((i * 7) ^ 8'ha5) & 255)) $fatal(1, "Loaded code[%0d]", i);
        run_program(8'h01);
        if (led[3:0] !== 4'b1101) $fatal(1, "Result LEDs");
        // A new start must clear done before the controller reads it.
        run_program(8'h82);
        if (dut.core.starts != 2) $fatal(1, "Start pulse count");
        fork send_byte(8'h58); expect_byte(8'h4b); join
        repeat (DIVISOR) @(negedge clk);
        if (dut.core.memory[0] !== 8'h82 || dut.core.done || dut.core.fault)
            $fatal(1, "Reset must clear flags while preserving RAM");
        load_program(0);
        ping();
        load_program(129);
        ping();
        fork send_byte(8'hff); expect_byte(8'h45); join
        $display("PASS top: ping, 128-instruction load, result packet, rerun, reset, invalid counts");
        $finish;
    end
    initial begin #20000000; $fatal(1, "Top protocol test timeout"); end
endmodule
