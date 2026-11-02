// UART bytecode loader and result packets for the SC8 ISA profile.
module top (
    input wire clk,
    input wire rx,
    output wire tx,
    output wire [4:0] led
);
    // iCE40 flip-flop initialization supplies a power-on reset without a pin.
    reg [4:0] power_count = 0;
    wire power_reset = !power_count[4];
    always @(posedge clk)
        if (power_reset) power_count <= power_count + 1'b1;

    wire [7:0] rx_data;
    wire rx_valid;
    reg [7:0] tx_data;
    reg tx_valid;
    wire tx_ready, tx_done;
    uart_rx receiver(.clk(clk), .reset(power_reset), .rx(rx),
                     .data(rx_data), .valid(rx_valid));
    uart_tx transmitter(.clk(clk), .reset(power_reset), .data(tx_data),
                        .valid(tx_valid), .ready(tx_ready), .tx(tx), .done(tx_done));

    reg core_reset, core_start;
    reg [15:0] step_limit;
    reg [7:0] program_length;
    reg host_write;
    reg [4:0] host_addr;
    reg [7:0] host_wdata;
    wire [7:0] host_rdata;
    reg code_write;
    reg [8:0] code_addr;
    reg [7:0] code_wdata;
    wire done, busy, carry;
    wire [7:0] fault, pc;
    wire [15:0] steps;
    isa_core core(
        .clk(clk), .reset(power_reset | core_reset), .start(core_start),
        .step_limit(step_limit), .program_length(program_length),
        .host_write(host_write), .host_addr(host_addr), .host_wdata(host_wdata),
        .host_rdata(host_rdata), .code_write(code_write), .code_addr(code_addr),
        .code_wdata(code_wdata), .done(done), .busy(busy), .fault(fault),
        .steps(steps), .pc(pc), .carry(carry)
    );
    // Red LEDs: carry, busy, done, fault. Green: power-on reset complete.
    assign led = {!power_reset, |fault, done, busy, carry};

    localparam COMMAND = 0, LOAD_MEMORY = 1, LOAD_LENGTH = 2, LOAD_CODE = 3,
               LIMIT_LOW = 4, LIMIT_HIGH = 5, RUN_START = 6, RUN_WAIT = 7,
               RESPONSE = 8, READ_WAIT1 = 9, READ_WAIT2 = 10,
               READ_SAMPLE = 11, TX_LAUNCH = 12, TX_WAIT = 13;
    localparam PING = 0, ACK = 1, ERROR = 2, RESULT = 3;
    reg [3:0] state;
    reg [1:0] response_kind;
    reg [5:0] response_index;
    reg [5:0] memory_index;
    reg [8:0] code_index, code_bytes;
    wire response_last = (response_kind == RESULT) ? response_index == 37 :
                         (response_kind == PING) ? response_index == 3 : 1'b1;

    always @(posedge clk) begin
        if (power_reset) begin
            state <= COMMAND;
            core_reset <= 0;
            core_start <= 0;
            step_limit <= 0;
            program_length <= 0;
            host_write <= 0;
            host_addr <= 0;
            host_wdata <= 0;
            code_write <= 0;
            code_addr <= 0;
            code_wdata <= 0;
            tx_data <= 0;
            tx_valid <= 0;
            response_kind <= ACK;
            response_index <= 0;
            memory_index <= 0;
            code_index <= 0;
            code_bytes <= 0;
        end else begin
            core_reset <= 0;
            core_start <= 0;
            host_write <= 0;
            code_write <= 0;
            tx_valid <= 0;
            case (state)
                COMMAND: if (rx_valid) begin
                    response_index <= 0;
                    case (rx_data)
                        8'h50: begin // P: protocol signature and version.
                            response_kind <= PING;
                            state <= RESPONSE;
                        end
                        8'h4c: begin // L: 32 memory bytes, count, then triples.
                            memory_index <= 0;
                            program_length <= 0;
                            state <= LOAD_MEMORY;
                        end
                        8'h52: state <= LIMIT_LOW; // R: little-endian step limit.
                        8'h58: begin // X: reset execution state, preserving RAM.
                            core_reset <= 1;
                            response_kind <= ACK;
                            state <= RESPONSE;
                        end
                        default: begin
                            response_kind <= ERROR;
                            state <= RESPONSE;
                        end
                    endcase
                end
                LOAD_MEMORY: if (rx_valid) begin
                    host_addr <= memory_index[4:0];
                    host_wdata <= rx_data;
                    host_write <= 1;
                    if (memory_index == 31) state <= LOAD_LENGTH;
                    else memory_index <= memory_index + 1'b1;
                end
                LOAD_LENGTH: if (rx_valid) begin
                    if (rx_data == 0 || rx_data > 128) begin
                        response_kind <= ERROR;
                        state <= RESPONSE;
                    end else begin
                        program_length <= rx_data;
                        code_bytes <= {1'b0, rx_data} + ({1'b0, rx_data} << 1);
                        code_index <= 0;
                        state <= LOAD_CODE;
                    end
                end
                LOAD_CODE: if (rx_valid) begin
                    code_addr <= code_index;
                    code_wdata <= rx_data;
                    code_write <= 1;
                    if (code_index == code_bytes - 1'b1) begin
                        response_kind <= ACK;
                        state <= RESPONSE;
                    end else code_index <= code_index + 1'b1;
                end
                LIMIT_LOW: if (rx_valid) begin
                    step_limit[7:0] <= rx_data;
                    state <= LIMIT_HIGH;
                end
                LIMIT_HIGH: if (rx_valid) begin
                    step_limit[15:8] <= rx_data;
                    core_start <= 1;
                    state <= RUN_START;
                end
                // Give start a complete cycle to clear the core's sticky done.
                RUN_START: state <= RUN_WAIT;
                RUN_WAIT: if (done) begin
                    response_kind <= RESULT;
                    response_index <= 0;
                    state <= RESPONSE;
                end
                RESPONSE: begin
                    if (response_kind == RESULT && response_index >= 6) begin
                        host_addr <= response_index - 6;
                        state <= READ_WAIT1;
                    end else begin
                        case (response_kind)
                            ACK: tx_data <= 8'h4b;
                            ERROR: tx_data <= 8'h45;
                            PING: case (response_index)
                                0: tx_data <= 8'h53;
                                1: tx_data <= 8'h43;
                                2: tx_data <= 8'h38;
                                default: tx_data <= 8'h01;
                            endcase
                            RESULT: case (response_index)
                                0: tx_data <= 8'h44;
                                1: tx_data <= fault;
                                2: tx_data <= steps[7:0];
                                3: tx_data <= steps[15:8];
                                4: tx_data <= pc;
                                default: tx_data <= {7'b0, carry};
                            endcase
                        endcase
                        state <= TX_LAUNCH;
                    end
                end
                READ_WAIT1: state <= READ_WAIT2;
                READ_WAIT2: state <= READ_SAMPLE;
                READ_SAMPLE: begin
                    tx_data <= host_rdata;
                    state <= TX_LAUNCH;
                end
                TX_LAUNCH: if (tx_ready) begin
                    tx_valid <= 1;
                    state <= TX_WAIT;
                end
                TX_WAIT: if (tx_done) begin
                    if (response_last) state <= COMMAND;
                    else begin
                        response_index <= response_index + 1'b1;
                        state <= RESPONSE;
                    end
                end
                default: state <= COMMAND;
            endcase
        end
    end
endmodule
