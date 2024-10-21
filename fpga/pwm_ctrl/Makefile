BOARD=tangnano9k
FAMILY=GW1N-9C
DEVICE=GW1NR-LV9QN88PC6/I5

all: pwm_ctrl.fs

# Synthesis
pwm_ctrl.json: top.v
	yosys -p "read_verilog top.v; synth_gowin -top top -json pwm_ctrl.json"

# Place and Route
pwm_ctrl_pnr.json: pwm_ctrl.json
	nextpnr-gowin --json pwm_ctrl.json --write pwm_ctrl_pnr.json --freq 27 --device ${DEVICE} --family ${FAMILY} --cst ${BOARD}.cst

# Generate Bitstream
pwm_ctrl.fs: pwm_ctrl_pnr.json
	gowin_pack -d ${FAMILY} -o pwm_ctrl.fs pwm_ctrl_pnr.json

# Program Board
load: pwm_ctrl.fs
	openFPGALoader -b ${BOARD} pwm_ctrl.fs -f

.PHONY: load
.INTERMEDIATE: pwm_ctrl_pnr.json pwm_ctrl.json
