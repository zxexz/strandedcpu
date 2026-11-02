// 8N1 UART: 104 clocks per bit at 12 MHz, about 115.4 kbaud.
module uart_rx #(
    parameter integer DIVISOR = 104
) (
    input wire clk,
    input wire reset,
    input wire rx,
    output reg [7:0] data,
    output reg valid
);
    reg rx_meta = 1'b1;
    reg rx_sync = 1'b1;
    reg [7:0] count;
    reg [2:0] bit_index;
    reg [7:0] shift;
    reg [1:0] state;
    localparam IDLE = 0, START = 1, DATA = 2, STOP = 3;

    always @(posedge clk) begin
        rx_meta <= rx;
        rx_sync <= rx_meta;
        if (reset) begin
            state <= IDLE;
            count <= 0;
            bit_index <= 0;
            shift <= 0;
            data <= 0;
            valid <= 0;
        end else begin
            valid <= 0;
            case (state)
                IDLE: if (!rx_sync) begin
                    count <= DIVISOR / 2 - 1;
                    state <= START;
                end
                START: if (count != 0) count <= count - 1'b1;
                else if (rx_sync) state <= IDLE;
                else begin
                    count <= DIVISOR - 1;
                    bit_index <= 0;
                    state <= DATA;
                end
                DATA: if (count != 0) count <= count - 1'b1;
                else begin
                    shift[bit_index] <= rx_sync;
                    count <= DIVISOR - 1;
                    if (bit_index == 7) state <= STOP;
                    else bit_index <= bit_index + 1'b1;
                end
                STOP: if (count != 0) count <= count - 1'b1;
                else begin
                    if (rx_sync) begin
                        data <= shift;
                        valid <= 1;
                    end
                    state <= IDLE;
                end
            endcase
        end
    end
endmodule

module uart_tx #(
    parameter integer DIVISOR = 104
) (
    input wire clk,
    input wire reset,
    input wire [7:0] data,
    input wire valid,
    output wire ready,
    output reg tx,
    output reg done
);
    reg busy;
    reg [7:0] count;
    reg [9:0] shift;
    reg [3:0] bits_left;
    assign ready = !busy;

    always @(posedge clk) begin
        if (reset) begin
            busy <= 0;
            count <= 0;
            shift <= 10'h3ff;
            bits_left <= 0;
            tx <= 1;
            done <= 0;
        end else begin
            done <= 0;
            if (!busy) begin
                if (valid) begin
                    shift <= {1'b1, data, 1'b0};
                    bits_left <= 10;
                    count <= DIVISOR - 1;
                    tx <= 0;
                    busy <= 1;
                end
            end else if (count != 0) count <= count - 1'b1;
            else if (bits_left == 1) begin
                busy <= 0;
                tx <= 1;
                done <= 1;
            end else begin
                tx <= shift[1];
                shift <= {1'b1, shift[9:1]};
                bits_left <= bits_left - 1'b1;
                count <= DIVISOR - 1;
            end
        end
    end
endmodule
