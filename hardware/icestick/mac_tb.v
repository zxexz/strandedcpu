`timescale 1ns/1ps
module mac_tb;
    localparam DIVISOR = 104;
    reg clk = 0;
    always #5 clk = !clk;
    reg rx = 1;
    wire tx;
    wire [4:0] led;
    mac_top dut(.clk(clk), .rx(rx), .tx(tx), .led(led));

    reg reset = 1, clear = 0, start = 0;
    reg signed [15:0] coefficient, activation;
    wire signed [31:0] accumulator;
    wire overflow, busy, done;
    mac_core unit(.clk(clk), .reset(reset), .clear(clear), .start(start),
                  .coefficient(coefficient), .activation(activation), .accumulator(accumulator),
                  .overflow(overflow), .busy(busy), .done(done));

    task reset_dot;
        begin
            @(negedge clk); clear = 1;
            @(negedge clk); clear = 0;
        end
    endtask
    task multiply(input signed [15:0] a, input signed [15:0] b);
        begin
            @(negedge clk); coefficient = a; activation = b; start = 1;
            @(negedge clk); start = 0;
            wait(done); @(negedge clk);
        end
    endtask
    task send_byte(input [7:0] value);
        integer j;
        begin
            @(negedge clk); rx = 0;
            repeat (DIVISOR) @(negedge clk);
            for (j = 0; j < 8; j = j + 1) begin
                rx = value[j]; repeat (DIVISOR) @(negedge clk);
            end
            rx = 1; repeat (DIVISOR) @(negedge clk);
        end
    endtask
    task expect_byte(input [7:0] expected);
        reg [7:0] value;
        integer j;
        begin
            @(negedge tx);
            repeat (DIVISOR / 2) @(posedge clk);
            #1;
            if (tx !== 0) $fatal(1, "MAC UART start");
            for (j = 0; j < 8; j = j + 1) begin
                repeat (DIVISOR) @(posedge clk); #1; value[j] = tx;
            end
            repeat (DIVISOR) @(posedge clk); #1;
            if (tx !== 1 || value !== expected)
                $fatal(1, "MAC UART got %02x expected %02x", value, expected);
        end
    endtask
    task begin_dot(input relu, input [15:0] count);
        begin
            send_byte("D"); send_byte({7'b0, relu});
            send_byte(count[7:0]); send_byte(count[15:8]);
        end
    endtask
    task pair(input signed [15:0] a, input signed [15:0] b);
        begin send_byte(a[7:0]); send_byte(a[15:8]); send_byte(b[7:0]); send_byte(b[15:8]); end
    endtask
    task expect_result(input [31:0] value, input overflowed);
        begin
            expect_byte("A"); expect_byte(value[7:0]); expect_byte(value[15:8]);
            expect_byte(value[23:16]); expect_byte(value[31:24]); expect_byte({7'b0, overflowed});
        end
    endtask

    integer i;
    reg signed [15:0] a, b;
    reg signed [31:0] expected_product;
    initial begin
        repeat (5) @(negedge clk); reset = 0;
        reset_dot(); multiply(-32768, -32768);
        if (accumulator !== 32'h40000000 || overflow) $fatal(1, "minimum magnitude product");
        multiply(-32768, -32768);
        if (accumulator !== 32'h80000000 || !overflow) $fatal(1, "positive intermediate overflow");
        multiply(0, 0);
        if (!overflow) $fatal(1, "overflow must be sticky");
        reset_dot(); multiply(-32768, 32767); multiply(-32768, 32767); multiply(-32768, 32767);
        if (!overflow) $fatal(1, "negative overflow");
        for (i = 0; i < 200; i = i + 1) begin
            a = $random; b = $random; expected_product = a * b;
            reset_dot(); multiply(a, b);
            if (accumulator !== expected_product || overflow)
                $fatal(1, "signed product %0d * %0d got %0d expected %0d", a, b, accumulator, expected_product);
        end
        $display("PASS MAC core: 200 signed products, min integer, both overflow directions, reset");
    end
    initial begin
        repeat (40) @(negedge clk);
        fork
            send_byte("P");
            begin expect_byte("S"); expect_byte("C"); expect_byte("M"); expect_byte("1"); end
        join
        repeat (DIVISOR) @(negedge clk);
        begin_dot(0, 2); pair(2, 3);
        fork pair(-4, 5); expect_result(-14, 0); join
        repeat (DIVISOR) @(negedge clk);
        begin_dot(1, 2); pair(2, 3);
        fork pair(-4, 5); expect_result(0, 0); join
        repeat (DIVISOR) @(negedge clk);
        begin_dot(0, 2); pair(-32768, -32768);
        fork pair(-32768, -32768); expect_result(32'h80000000, 1); join
        repeat (DIVISOR) @(negedge clk);
        begin_dot(0, 1);
        fork pair(-32768, 32767); expect_result(-1073709056, 0); join
        repeat (DIVISOR) @(negedge clk);
        send_byte("D"); send_byte(0); send_byte(0);
        fork send_byte(0); expect_result(0, 0); join
        $display("PASS MAC UART: identity, signed dot, ReLU, overflow response, reset, zero terms");
        $finish;
    end
    initial begin #20000000; $fatal(1, "MAC test timeout"); end
endmodule
