YOSYS ?= yosys
NEXTPNR ?= nextpnr-ice40
ICEPACK ?= icepack
IVERILOG ?= iverilog
VVP ?= vvp
BUILD_DIR ?= build

.PHONY: all mac clean test test-uart test-top test-integration test-mac
all: $(BUILD_DIR)/top.bin
mac: $(BUILD_DIR)/mac.bin

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/top.json: top.v uart.v isa_core.v | $(BUILD_DIR)
	$(YOSYS) -l $(BUILD_DIR)/yosys.log -p 'synth_ice40 -top top -json $@' uart.v isa_core.v top.v

$(BUILD_DIR)/top.asc: $(BUILD_DIR)/top.json icestick.pcf
	$(NEXTPNR) --hx1k --package tq144 --freq 12 --json $< --pcf icestick.pcf --asc $@ --report $(BUILD_DIR)/timing.json

$(BUILD_DIR)/top.bin: $(BUILD_DIR)/top.asc
	$(ICEPACK) $< $@

$(BUILD_DIR)/mac.json: mac_top.v mac_core.v uart.v | $(BUILD_DIR)
	$(YOSYS) -l $(BUILD_DIR)/mac-yosys.log -p 'synth_ice40 -top mac_top -json $@' uart.v mac_core.v mac_top.v

$(BUILD_DIR)/mac.asc: $(BUILD_DIR)/mac.json icestick.pcf
	$(NEXTPNR) --hx1k --package tq144 --freq 12 --json $< --pcf icestick.pcf --asc $@ --report $(BUILD_DIR)/mac-timing.json

$(BUILD_DIR)/mac.bin: $(BUILD_DIR)/mac.asc
	$(ICEPACK) $< $@

test: test-uart test-top test-integration

test-uart: | $(BUILD_DIR)
	$(IVERILOG) -g2012 -s uart_tb -o $(BUILD_DIR)/uart_tb tests/uart_tb.v uart.v
	$(VVP) $(BUILD_DIR)/uart_tb

test-top: | $(BUILD_DIR)
	$(IVERILOG) -g2012 -s top_tb -o $(BUILD_DIR)/top_tb tests/top_tb.v uart.v top.v
	$(VVP) $(BUILD_DIR)/top_tb

test-integration: | $(BUILD_DIR)
	$(IVERILOG) -g2012 -s integration_tb -o $(BUILD_DIR)/integration_tb tests/integration_tb.v uart.v isa_core.v top.v
	$(VVP) $(BUILD_DIR)/integration_tb

test-mac: | $(BUILD_DIR)
	$(IVERILOG) -g2012 -s mac_tb -o $(BUILD_DIR)/mac_tb mac_tb.v uart.v mac_core.v mac_top.v
	$(VVP) $(BUILD_DIR)/mac_tb

clean:
	rm -rf $(BUILD_DIR)
