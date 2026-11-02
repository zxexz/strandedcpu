// Signed 16-bit shift-add products with a 32-bit accumulator and overflow flag.
module mac_core (
    input wire clk,
    input wire reset,
    input wire clear,
    input wire start,
    input wire signed [15:0] coefficient,
    input wire signed [15:0] activation,
    output reg signed [31:0] accumulator,
    output reg overflow,
    output reg busy,
    output reg done
);
    reg [31:0] multiplicand;
    reg [15:0] multiplier;
    reg [31:0] product;
    reg negative;
    reg [3:0] bit_count;

    wire [31:0] next_product = product + (multiplier[0] ? multiplicand : 32'b0);
    wire [31:0] signed_product = negative ? (~next_product + 1'b1) : next_product;
    wire [32:0] next_sum = {accumulator[31], accumulator} +
                          {signed_product[31], signed_product};

    always @(posedge clk) begin
        if (reset || clear) begin
            accumulator <= 0;
            overflow <= 0;
            busy <= 0;
            done <= 0;
            multiplicand <= 0;
            multiplier <= 0;
            product <= 0;
            negative <= 0;
            bit_count <= 0;
        end else begin
            done <= 0;
            if (start && !busy) begin
                // Unsigned magnitude 32768 represents abs(-32768) without overflow.
                multiplicand <= {16'b0, coefficient[15] ? (~coefficient + 16'd1) : coefficient};
                multiplier <= activation[15] ? (~activation + 16'd1) : activation;
                product <= 0;
                negative <= coefficient[15] ^ activation[15];
                bit_count <= 0;
                busy <= 1;
            end else if (busy) begin
                if (bit_count == 15) begin
                    accumulator <= next_sum[31:0];
                    overflow <= overflow | (next_sum[32] != next_sum[31]);
                    busy <= 0;
                    done <= 1;
                end else begin
                    product <= next_product;
                    multiplicand <= multiplicand << 1;
                    multiplier <= multiplier >> 1;
                    bit_count <= bit_count + 1'b1;
                end
            end
        end
    end
endmodule
