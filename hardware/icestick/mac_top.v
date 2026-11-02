// SCM1 UART interface for streamed dot products.
module mac_top (
    input wire clk,
    input wire rx,
    output wire tx,
    output wire [4:0] led
);
    reg [4:0] power_count = 0;
    wire power_reset = !power_count[4];
    always @(posedge clk)
        if (power_reset) power_count <= power_count + 1'b1;

    wire [7:0] rx_data;
    wire rx_valid;
    reg [7:0] tx_data;
    reg tx_valid;
    wire tx_ready, tx_done;
    uart_rx receiver(.clk(clk), .reset(power_reset), .rx(rx), .data(rx_data), .valid(rx_valid));
    uart_tx transmitter(.clk(clk), .reset(power_reset), .data(tx_data), .valid(tx_valid),
                        .ready(tx_ready), .tx(tx), .done(tx_done));

    reg core_clear, core_start;
    reg signed [15:0] coefficient, activation;
    wire signed [31:0] accumulator;
    wire overflow, busy, done;
    mac_core core(.clk(clk), .reset(power_reset), .clear(core_clear), .start(core_start),
                  .coefficient(coefficient), .activation(activation), .accumulator(accumulator),
                  .overflow(overflow), .busy(busy), .done(done));

    localparam COMMAND=0, FLAGS=1, COUNT_LOW=2, COUNT_HIGH=3,
               A_LOW=4, A_HIGH=5, B_LOW=6, B_HIGH=7, MAC_WAIT=8,
               RESPONSE=9, TX_LAUNCH=10, TX_WAIT=11;
    reg [3:0] state;
    reg [15:0] remaining;
    reg relu;
    reg [31:0] reply_result;
    reg reply_overflow;
    reg [1:0] reply_kind;
    reg [2:0] reply_index;
    localparam PING=0, RESULT=1, ERROR=2;
    wire reply_last = reply_kind == PING ? reply_index == 3 :
                      reply_kind == RESULT ? reply_index == 5 : 1'b1;
    assign led = {!power_reset, overflow, busy, state != COMMAND, 1'b0};

    always @(posedge clk) begin
        if (power_reset) begin
            state <= COMMAND;
            remaining <= 0;
            relu <= 0;
            coefficient <= 0;
            activation <= 0;
            core_clear <= 0;
            core_start <= 0;
            reply_result <= 0;
            reply_overflow <= 0;
            reply_kind <= PING;
            reply_index <= 0;
            tx_data <= 0;
            tx_valid <= 0;
        end else begin
            core_clear <= 0;
            core_start <= 0;
            tx_valid <= 0;
            case (state)
                COMMAND: if (rx_valid) begin
                    reply_index <= 0;
                    case (rx_data)
                        8'h50: begin reply_kind <= PING; state <= RESPONSE; end
                        8'h44: begin core_clear <= 1; state <= FLAGS; end
                        default: begin reply_kind <= ERROR; state <= RESPONSE; end
                    endcase
                end
                FLAGS: if (rx_valid) begin relu <= rx_data[0]; state <= COUNT_LOW; end
                COUNT_LOW: if (rx_valid) begin remaining[7:0] <= rx_data; state <= COUNT_HIGH; end
                COUNT_HIGH: if (rx_valid) begin
                    remaining[15:8] <= rx_data;
                    if ({rx_data, remaining[7:0]} == 0) begin
                        reply_kind <= RESULT;
                        reply_result <= 0;
                        reply_overflow <= 0;
                        state <= RESPONSE;
                    end else state <= A_LOW;
                end
                A_LOW: if (rx_valid) begin coefficient[7:0] <= rx_data; state <= A_HIGH; end
                A_HIGH: if (rx_valid) begin coefficient[15:8] <= rx_data; state <= B_LOW; end
                B_LOW: if (rx_valid) begin activation[7:0] <= rx_data; state <= B_HIGH; end
                B_HIGH: if (rx_valid) begin
                    activation[15:8] <= rx_data;
                    core_start <= 1;
                    state <= MAC_WAIT;
                end
                MAC_WAIT: if (done) begin
                    if (remaining == 1) begin
                        reply_result <= relu && accumulator[31] ? 32'b0 : accumulator;
                        reply_overflow <= overflow;
                        reply_kind <= RESULT;
                        state <= RESPONSE;
                    end else begin
                        remaining <= remaining - 1'b1;
                        state <= A_LOW;
                    end
                end
                RESPONSE: begin
                    case (reply_kind)
                        PING: case (reply_index)
                            0: tx_data <= 8'h53;
                            1: tx_data <= 8'h43;
                            2: tx_data <= 8'h4d;
                            default: tx_data <= 8'h31;
                        endcase
                        RESULT: case (reply_index)
                            0: tx_data <= 8'h41;
                            1: tx_data <= reply_result[7:0];
                            2: tx_data <= reply_result[15:8];
                            3: tx_data <= reply_result[23:16];
                            4: tx_data <= reply_result[31:24];
                            default: tx_data <= {7'b0, reply_overflow};
                        endcase
                        default: tx_data <= 8'h45;
                    endcase
                    state <= TX_LAUNCH;
                end
                TX_LAUNCH: if (tx_ready) begin tx_valid <= 1; state <= TX_WAIT; end
                TX_WAIT: if (tx_done) begin
                    if (reply_last) state <= COMMAND;
                    else begin reply_index <= reply_index + 1'b1; state <= RESPONSE; end
                end
                default: state <= COMMAND;
            endcase
        end
    end
endmodule
