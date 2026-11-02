`timescale 1ns/1ps
module uart_tb;
    localparam DIVISOR = 104;
    reg clk = 0;
    always #5 clk = !clk;
    reg reset = 1;
    reg rx = 1;
    wire [7:0] rx_data;
    wire rx_valid;
    reg [7:0] tx_data;
    reg tx_valid = 0;
    wire tx_ready, tx, tx_done;
    integer received = 0;
    reg [7:0] captured [0:7];
    uart_rx receiver(.clk(clk), .reset(reset), .rx(rx), .data(rx_data), .valid(rx_valid));
    uart_tx transmitter(.clk(clk), .reset(reset), .data(tx_data), .valid(tx_valid),
                        .ready(tx_ready), .tx(tx), .done(tx_done));

    always @(posedge clk) if (rx_valid) begin
        captured[received] = rx_data;
        received = received + 1;
    end

    task send_rx(input [7:0] value, input stop_bit);
        integer bit_number;
        begin
            @(negedge clk); rx = 0;
            repeat (DIVISOR) @(negedge clk);
            for (bit_number = 0; bit_number < 8; bit_number = bit_number + 1) begin
                rx = value[bit_number];
                repeat (DIVISOR) @(negedge clk);
            end
            rx = stop_bit;
            repeat (DIVISOR) @(negedge clk);
            rx = 1;
        end
    endtask

    task launch_tx(input [7:0] value);
        begin
            wait (tx_ready);
            @(negedge clk); tx_data = value; tx_valid = 1;
            @(negedge clk); tx_valid = 0;
        end
    endtask

    task expect_tx(input [7:0] value);
        integer bit_number;
        begin
            @(negedge tx);
            repeat (DIVISOR / 2) @(posedge clk);
            #1;
            if (tx !== 0 || tx_ready !== 0) $fatal(1, "TX start/ready");
            for (bit_number = 0; bit_number < 8; bit_number = bit_number + 1) begin
                repeat (DIVISOR) @(posedge clk);
                #1;
                if (tx !== value[bit_number]) $fatal(1, "TX data bit %0d", bit_number);
            end
            repeat (DIVISOR) @(posedge clk);
            #1;
            if (tx !== 1 || tx_ready !== 0) $fatal(1, "TX stop/ready");
            wait (tx_done);
            #1;
            if (!tx_ready || !tx) $fatal(1, "TX completion");
        end
    endtask

    initial begin
        repeat (5) @(negedge clk);
        reset = 0;
        repeat (5) @(negedge clk);
        if (!tx || !tx_ready) $fatal(1, "TX idle");
        // Reject pulses shorter than half a bit.
        rx = 0;
        repeat (2) @(negedge clk);
        rx = 1;
        repeat (DIVISOR) @(negedge clk);
        if (received != 0) $fatal(1, "RX accepted a start glitch");
        send_rx(8'h00, 1);
        send_rx(8'hff, 1);
        send_rx(8'ha6, 1);
        repeat (5) @(negedge clk);
        if (received != 3 || captured[0] !== 8'h00 || captured[1] !== 8'hff || captured[2] !== 8'ha6)
            $fatal(1, "RX byte sequence");
        send_rx(8'h55, 0);
        repeat (DIVISOR) @(negedge clk);
        if (received != 3) $fatal(1, "RX accepted an invalid stop bit");
        fork launch_tx(8'ha6); expect_tx(8'ha6); join
        fork launch_tx(8'h00); expect_tx(8'h00); join
        fork launch_tx(8'hff); expect_tx(8'hff); join
        $display("PASS UART: framing, synchronization, byte order, TX handshake");
        $finish;
    end
    initial begin #200000; $fatal(1, "UART test timeout"); end
endmodule
