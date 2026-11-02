// Multicycle ISA executor: N=8, s=32, m=32, up to 128 three-byte instructions.
module isa_core (
    input wire clk,
    input wire reset,
    input wire start,
    input wire [15:0] step_limit,
    input wire [7:0] program_length,
    input wire host_write,
    input wire [4:0] host_addr,
    input wire [7:0] host_wdata,
    output wire [7:0] host_rdata,
    input wire code_write,
    input wire [8:0] code_addr,
    input wire [7:0] code_wdata,
    output reg done,
    output reg busy,
    output reg [7:0] fault,
    output reg [15:0] steps,
    output reg [7:0] pc,
    output reg carry
);
    // Byte-wide instruction storage uses one RAM4K. Block RAM for the
    // 32-word data memory saves LUTs despite leaving most of that RAM unused.
    (* ram_style = "block" *) reg [7:0] code_mem [0:511];
    (* ram_style = "block" *) reg [7:0] data_mem [0:31];
    reg [7:0] code_q, data_q;
    reg [8:0] fetch_addr;
    reg [4:0] data_addr;
    reg data_we;
    reg [7:0] data_wdata;

    localparam IDLE=0, FETCH=1, FETCH_WAIT=2, FETCH_LATCH=3,
               DECODE=4, READ_B=5, LATCH_B=6, READ_C=7, LATCH_C=8,
               READ_X=9, LATCH_X=10, EXECUTE=11, WRITE_0=12,
               WRITE_1=13, RETIRE=14;
    reg [3:0] state;
    reg [1:0] fetch_part;
    reg [7:0] op, b, c, vb, vc, vx;
    reg [7:0] next_pc;
    reg [4:0] write_addr_0, write_addr_1;
    reg [7:0] write_data_0, write_data_1;
    reg write_twice;

    wire subleq = op >= 8'd32 && op <= 8'd63;
    wire supported = op <= 8'd15 ||
                     (op >= 8'd17 && op <= 8'd21) ||
                     (op >= 8'd24 && op <= 8'd30) ||
                     subleq || op == 8'd64;
    // Only fields used as data addresses require the 32..63 range.
    wire needs_b = subleq || (op != 0 && op != 3 && op != 64);
    wire needs_c = subleq ? 1'b0 :
                   (op == 1 || op == 2 || op == 11 || op == 12 ||
                    op == 13 || op == 14 || op == 15 ||
                    (op >= 17 && op <= 21) ||
                    (op >= 24 && op <= 30) || op == 64);
    wire b_valid = b >= 8'd32 && b < 8'd64;
    wire c_valid = c >= 8'd32 && c < 8'd64;
    wire [8:0] add_value = {1'b0, vb} + {1'b0, vc};
    wire [8:0] adc_value = add_value + {8'd0, carry};
    wire [7:0] sub_value = vb - vc;
    wire [7:0] subleq_value = vb - vx;
    wire [8:0] multiply_value = {1'b0, vb} + (vx[0] ? {1'b0, vc} : 9'd0);
    wire [8:0] divide_value = {vb, vx[7]};
    wire divide_ge = divide_value >= {1'b0, vc};
    wire [8:0] divide_result = divide_ge ? divide_value - {1'b0, vc} : divide_value;
    wire [7:0] right_amount = (~vc) + 8'd1;

    function [7:0] leading_zeros;
        input [7:0] value;
        begin
            casez (value)
                8'b1???????: leading_zeros = 0;
                8'b01??????: leading_zeros = 1;
                8'b001?????: leading_zeros = 2;
                8'b0001????: leading_zeros = 3;
                8'b00001???: leading_zeros = 4;
                8'b000001??: leading_zeros = 5;
                8'b0000001?: leading_zeros = 6;
                8'b00000001: leading_zeros = 7;
                default: leading_zeros = 8;
            endcase
        end
    endfunction

    // The host owns the RAM port while idle. Separate address and latch
    // states account for synchronous operand reads.
    always @* begin
        data_addr = host_addr;
        data_we = host_write && !busy;
        data_wdata = host_wdata;
        if (busy) begin
            data_addr = 0;
            data_we = 0;
            data_wdata = 0;
            case (state)
                READ_B, LATCH_B: data_addr = b[4:0];
                READ_C, LATCH_C: data_addr = c[4:0];
                READ_X, LATCH_X: begin
                    if (subleq) data_addr = op[4:0];
                    else if (op == 26 || op == 27) data_addr = b[4:0] ^ 5'd1;
                    else if (op == 11) data_addr = vc[4:0];
                end
                WRITE_0: begin
                    data_addr = write_addr_0;
                    data_wdata = write_data_0;
                    data_we = 1;
                end
                WRITE_1: begin
                    data_addr = write_addr_1;
                    data_wdata = write_data_1;
                    data_we = 1;
                end
            endcase
        end
    end

    always @(posedge clk) begin
        if (code_write && !busy) code_mem[code_addr] <= code_wdata;
        code_q <= code_mem[busy ? fetch_addr : code_addr];
        if (data_we) data_mem[data_addr] <= data_wdata;
        data_q <= data_mem[data_addr];
    end
    assign host_rdata = data_q;

    always @(posedge clk) begin
        if (reset) begin
            state <= IDLE;
            done <= 0;
            busy <= 0;
            fault <= 0;
            steps <= 0;
            pc <= 0;
            carry <= 0;
            fetch_addr <= 0;
            fetch_part <= 0;
            op <= 0;
            b <= 0;
            c <= 0;
            vb <= 0;
            vc <= 0;
            vx <= 0;
            next_pc <= 0;
            write_addr_0 <= 0;
            write_addr_1 <= 0;
            write_data_0 <= 0;
            write_data_1 <= 0;
            write_twice <= 0;
        end else if (start && !busy) begin
            done <= 0;
            busy <= 1;
            fault <= 0;
            steps <= 0;
            pc <= 64;
            carry <= 0;
            state <= FETCH;
        end else begin
            case (state)
                IDLE: begin end
                FETCH: begin
                    if (pc == 0) begin
                        done <= 1;
                        busy <= 0;
                        state <= IDLE;
                    end else if (program_length == 0 || program_length > 128 ||
                                 pc < 64 || (pc - 8'd64) >= program_length) begin
                        fault <= 2;
                        done <= 1;
                        busy <= 0;
                        state <= IDLE;
                    end else if (steps >= step_limit) begin
                        fault <= 3;
                        done <= 1;
                        busy <= 0;
                        state <= IDLE;
                    end else begin
                        fetch_addr <= ({1'b0, pc} - 9'd64) * 9'd3;
                        fetch_part <= 0;
                        state <= FETCH_WAIT;
                    end
                end
                FETCH_WAIT: state <= FETCH_LATCH;
                FETCH_LATCH: begin
                    case (fetch_part)
                        0: op <= code_q;
                        1: b <= code_q;
                        2: c <= code_q;
                    endcase
                    if (fetch_part == 2) state <= DECODE;
                    else begin
                        fetch_part <= fetch_part + 1'b1;
                        fetch_addr <= fetch_addr + 1'b1;
                        state <= FETCH_WAIT;
                    end
                end
                DECODE: begin
                    if (!supported || (needs_b && !b_valid) || (needs_c && !c_valid)) begin
                        fault <= supported ? 8'd2 : 8'd1;
                        done <= 1;
                        busy <= 0;
                        state <= IDLE;
                    end else state <= READ_B;
                end
                READ_B: state <= LATCH_B;
                LATCH_B: begin vb <= data_q; state <= READ_C; end
                READ_C: state <= LATCH_C;
                LATCH_C: begin vc <= data_q; state <= READ_X; end
                READ_X: state <= LATCH_X;
                LATCH_X: begin vx <= data_q; state <= EXECUTE; end
                EXECUTE: begin
                    next_pc <= pc + 1'b1;
                    write_addr_0 <= b[4:0];
                    write_addr_1 <= b[4:0] ^ 5'd1;
                    write_twice <= 0;
                    state <= WRITE_0;
                    if (subleq) begin
                        write_data_0 <= subleq_value;
                        if (subleq_value[7] || subleq_value == 0) next_pc <= c;
                    end else case (op)
                        0: begin next_pc <= 0; state <= RETIRE; end
                        1: write_data_0 <= vc;
                        2: begin write_data_0 <= add_value[7:0]; carry <= add_value[8]; end
                        3: begin next_pc <= c; state <= RETIRE; end
                        4: begin if (vb == 0) next_pc <= c; state <= RETIRE; end
                        5: begin if (vb != 0) next_pc <= c; state <= RETIRE; end
                        6: write_data_0 <= vb + 1'b1;
                        7: write_data_0 <= vb - 1'b1;
                        8: write_data_0 <= {vb[6:0], 1'b0};
                        9: write_data_0 <= {vb[7], vb[7:1]};
                        10: begin if (vb[7]) next_pc <= c; state <= RETIRE; end
                        11: write_data_0 <= vx;
                        12: write_data_0 <= vb & vc;
                        13: write_data_0 <= vb | vc;
                        14: write_data_0 <= vb ^ vc;
                        15: begin write_data_0 <= sub_value; carry <= vb >= vc; end
                        17: begin
                            write_data_0 <= vc;
                            write_addr_1 <= c[4:0];
                            write_data_1 <= vb;
                            write_twice <= 1;
                        end
                        18: begin write_data_0 <= vc; if (!vb[7]) state <= RETIRE; end
                        19: write_data_0 <= {vb[6:0], 1'b0} + (vb[7] ? vc : 8'd0);
                        20: begin write_addr_0 <= vc[4:0]; write_data_0 <= vb; end
                        21: write_data_0 <= $signed(vb) >= $signed(vc) ? vb : vc;
                        24: begin
                            if (!vc[7]) write_data_0 <= vc >= 8 ? 8'd0 : vb << vc[2:0];
                            else if (right_amount >= 8) write_data_0 <= vb[7] ? 8'hff : 8'd0;
                            else write_data_0 <= $signed(vb) >>> right_amount[2:0];
                        end
                        25: begin write_data_0 <= adc_value[7:0]; carry <= adc_value[8]; end
                        26: begin
                            write_data_0 <= multiply_value[8:1];
                            write_data_1 <= {multiply_value[0], vx[7:1]};
                            write_twice <= 1;
                        end
                        27: begin
                            write_data_0 <= divide_result[7:0];
                            write_data_1 <= {vx[6:0], divide_ge};
                            write_twice <= 1;
                        end
                        28: write_data_0 <= {1'b0, vb[7:1]} ^ (vb[0] ? vc : 8'd0);
                        29: write_data_0 <= leading_zeros(vc);
                        30: write_data_0 <= vc[7] ? -vc : vc;
                        64: begin
                            state <= RETIRE;
                            // A return offset cannot wrap the 8-bit PC.
                            if (vc >= program_length) begin
                                fault <= 2;
                                done <= 1;
                                busy <= 0;
                                state <= IDLE;
                            end else next_pc <= 8'd64 + vc;
                        end
                    endcase
                end
                WRITE_0: state <= write_twice ? WRITE_1 : RETIRE;
                WRITE_1: state <= RETIRE;
                RETIRE: begin
                    steps <= steps + 1'b1;
                    pc <= next_pc;
                    if (next_pc == 0) begin
                        done <= 1;
                        busy <= 0;
                        state <= IDLE;
                    end else state <= FETCH;
                end
                default: state <= IDLE;
            endcase
        end
    end
endmodule
