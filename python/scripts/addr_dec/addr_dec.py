# Copyright 2024
# Bryce's Senior Project
# Description: This script parses the register list and generates some verilog code

import re

str = ""
with open("D:/Senior_Project_Local/Senior_Project/python/scripts/addr_dec/regs.txt", 'r') as file:
    for line in file:
        # str += line.strip())  # strip() removes the newline character
        result = re.match(r"0x[0]*([0-9A-F]+)\t(.*)\t\t.*", line)

        if(result):
            # str += "[1] = " + result.group(1) + ", [2] = " + result.group(2))
            reg_name = re.match(r"([A-Z]+)([0-9])_.*", result.group(2))

            if((reg_name.group(1) == "DRIVE") | (reg_name.group(1) == "ROTATION")):
                str += "// ------------- 0x" + result.group(1) + "\t" + result.group(2) + "-------------\n"
                str += "always @(posedge clock) begin" + "\n"
                str += "\tif(write_en & (address == 6'h" + result.group(1) + "))" + "\n"
                str += "\t\treg_file[" + result.group(1) + "]     <=  wr_data[7:0];" + "\n"
                str += "end" + "\n\n"
                str += "assign brake" + reg_name.group(2) + "       = reg_file[" + result.group(1) + "][7];" + "\n"
                str += "assign enable" + reg_name.group(2) + "      = reg_file[" + result.group(1) + "][6];" + "\n"
                str += "assign direction" + reg_name.group(2) + "   = reg_file[" + result.group(1) + "][5];" + "\n"
                str += "assign pwm" + reg_name.group(2) + "[4:0]    = reg_file[" + result.group(1) + "][4:0];" + "\n"
                str += "\n"

# Open a file in write mode
with open('code.v', 'w') as file:
    # Write a string to the file
    file.write(str)
