// ------------- 0x4	DRIVE0_CONTROL	-------------
always @(posedge clock) begin
	if(write_en & (address == 6'h4))
		reg_file[4]     <=  wr_data[7:0];
end

assign brake0       = reg_file[4][7];
assign enable0      = reg_file[4][6];
assign direction0   = reg_file[4][5];
assign pwm0[4:0]    = reg_file[4][4:0];

// ------------- 0x5	DRIVE0_STATUS	-------------
always @(posedge clock) begin
	if(write_en & (address == 6'h5))
		reg_file[5]     <=  wr_data[7:0];
end

assign brake0       = reg_file[5][7];
assign enable0      = reg_file[5][6];
assign direction0   = reg_file[5][5];
assign pwm0[4:0]    = reg_file[5][4:0];

// ------------- 0x6	DRIVE1_CONTROL	-------------
always @(posedge clock) begin
	if(write_en & (address == 6'h6))
		reg_file[6]     <=  wr_data[7:0];
end

assign brake1       = reg_file[6][7];
assign enable1      = reg_file[6][6];
assign direction1   = reg_file[6][5];
assign pwm1[4:0]    = reg_file[6][4:0];

// ------------- 0x7	DRIVE1_STATUS	-------------
always @(posedge clock) begin
	if(write_en & (address == 6'h7))
		reg_file[7]     <=  wr_data[7:0];
end

assign brake1       = reg_file[7][7];
assign enable1      = reg_file[7][6];
assign direction1   = reg_file[7][5];
assign pwm1[4:0]    = reg_file[7][4:0];

// ------------- 0x8	DRIVE2_CONTROL	-------------
always @(posedge clock) begin
	if(write_en & (address == 6'h8))
		reg_file[8]     <=  wr_data[7:0];
end

assign brake2       = reg_file[8][7];
assign enable2      = reg_file[8][6];
assign direction2   = reg_file[8][5];
assign pwm2[4:0]    = reg_file[8][4:0];

// ------------- 0x9	DRIVE2_STATUS	-------------
always @(posedge clock) begin
	if(write_en & (address == 6'h9))
		reg_file[9]     <=  wr_data[7:0];
end

assign brake2       = reg_file[9][7];
assign enable2      = reg_file[9][6];
assign direction2   = reg_file[9][5];
assign pwm2[4:0]    = reg_file[9][4:0];

// ------------- 0xA	DRIVE3_CONTROL	-------------
always @(posedge clock) begin
	if(write_en & (address == 6'hA))
		reg_file[A]     <=  wr_data[7:0];
end

assign brake3       = reg_file[A][7];
assign enable3      = reg_file[A][6];
assign direction3   = reg_file[A][5];
assign pwm3[4:0]    = reg_file[A][4:0];

// ------------- 0xB	DRIVE3_STATUS	-------------
always @(posedge clock) begin
	if(write_en & (address == 6'hB))
		reg_file[B]     <=  wr_data[7:0];
end

assign brake3       = reg_file[B][7];
assign enable3      = reg_file[B][6];
assign direction3   = reg_file[B][5];
assign pwm3[4:0]    = reg_file[B][4:0];

// ------------- 0xC	ROTATION0_CONTROL	-------------
always @(posedge clock) begin
	if(write_en & (address == 6'hC))
		reg_file[C]     <=  wr_data[7:0];
end

assign brake0       = reg_file[C][7];
assign enable0      = reg_file[C][6];
assign direction0   = reg_file[C][5];
assign pwm0[4:0]    = reg_file[C][4:0];

// ------------- 0xD	ROTATION0_STATUS	-------------
always @(posedge clock) begin
	if(write_en & (address == 6'hD))
		reg_file[D]     <=  wr_data[7:0];
end

assign brake0       = reg_file[D][7];
assign enable0      = reg_file[D][6];
assign direction0   = reg_file[D][5];
assign pwm0[4:0]    = reg_file[D][4:0];

// ------------- 0xE	ROTATION0_TARG_ANG	-------------
always @(posedge clock) begin
	if(write_en & (address == 6'hE))
		reg_file[E]     <=  wr_data[7:0];
end

assign brake0       = reg_file[E][7];
assign enable0      = reg_file[E][6];
assign direction0   = reg_file[E][5];
assign pwm0[4:0]    = reg_file[E][4:0];

// ------------- 0xF	ROTATION0_CURR_ANG	-------------
always @(posedge clock) begin
	if(write_en & (address == 6'hF))
		reg_file[F]     <=  wr_data[7:0];
end

assign brake0       = reg_file[F][7];
assign enable0      = reg_file[F][6];
assign direction0   = reg_file[F][5];
assign pwm0[4:0]    = reg_file[F][4:0];

// ------------- 0x10	ROTATION1_CONTROL	-------------
always @(posedge clock) begin
	if(write_en & (address == 6'h10))
		reg_file[10]     <=  wr_data[7:0];
end

assign brake1       = reg_file[10][7];
assign enable1      = reg_file[10][6];
assign direction1   = reg_file[10][5];
assign pwm1[4:0]    = reg_file[10][4:0];

// ------------- 0x11	ROTATION1_STATUS	-------------
always @(posedge clock) begin
	if(write_en & (address == 6'h11))
		reg_file[11]     <=  wr_data[7:0];
end

assign brake1       = reg_file[11][7];
assign enable1      = reg_file[11][6];
assign direction1   = reg_file[11][5];
assign pwm1[4:0]    = reg_file[11][4:0];

// ------------- 0x12	ROTATION1_TARG_ANG	-------------
always @(posedge clock) begin
	if(write_en & (address == 6'h12))
		reg_file[12]     <=  wr_data[7:0];
end

assign brake1       = reg_file[12][7];
assign enable1      = reg_file[12][6];
assign direction1   = reg_file[12][5];
assign pwm1[4:0]    = reg_file[12][4:0];

// ------------- 0x13	ROTATION1_CURR_ANG	-------------
always @(posedge clock) begin
	if(write_en & (address == 6'h13))
		reg_file[13]     <=  wr_data[7:0];
end

assign brake1       = reg_file[13][7];
assign enable1      = reg_file[13][6];
assign direction1   = reg_file[13][5];
assign pwm1[4:0]    = reg_file[13][4:0];

// ------------- 0x14	ROTATION2_CONTROL	-------------
always @(posedge clock) begin
	if(write_en & (address == 6'h14))
		reg_file[14]     <=  wr_data[7:0];
end

assign brake2       = reg_file[14][7];
assign enable2      = reg_file[14][6];
assign direction2   = reg_file[14][5];
assign pwm2[4:0]    = reg_file[14][4:0];

// ------------- 0x15	ROTATION2_STATUS	-------------
always @(posedge clock) begin
	if(write_en & (address == 6'h15))
		reg_file[15]     <=  wr_data[7:0];
end

assign brake2       = reg_file[15][7];
assign enable2      = reg_file[15][6];
assign direction2   = reg_file[15][5];
assign pwm2[4:0]    = reg_file[15][4:0];

// ------------- 0x16	ROTATION2_TARG_ANG	-------------
always @(posedge clock) begin
	if(write_en & (address == 6'h16))
		reg_file[16]     <=  wr_data[7:0];
end

assign brake2       = reg_file[16][7];
assign enable2      = reg_file[16][6];
assign direction2   = reg_file[16][5];
assign pwm2[4:0]    = reg_file[16][4:0];

// ------------- 0x17	ROTATION2_CURR_ANG	-------------
always @(posedge clock) begin
	if(write_en & (address == 6'h17))
		reg_file[17]     <=  wr_data[7:0];
end

assign brake2       = reg_file[17][7];
assign enable2      = reg_file[17][6];
assign direction2   = reg_file[17][5];
assign pwm2[4:0]    = reg_file[17][4:0];

// ------------- 0x18	ROTATION3_CONTROL	-------------
always @(posedge clock) begin
	if(write_en & (address == 6'h18))
		reg_file[18]     <=  wr_data[7:0];
end

assign brake3       = reg_file[18][7];
assign enable3      = reg_file[18][6];
assign direction3   = reg_file[18][5];
assign pwm3[4:0]    = reg_file[18][4:0];

// ------------- 0x19	ROTATION3_STATUS	-------------
always @(posedge clock) begin
	if(write_en & (address == 6'h19))
		reg_file[19]     <=  wr_data[7:0];
end

assign brake3       = reg_file[19][7];
assign enable3      = reg_file[19][6];
assign direction3   = reg_file[19][5];
assign pwm3[4:0]    = reg_file[19][4:0];

// ------------- 0x1A	ROTATION3_TARG_ANG	-------------
always @(posedge clock) begin
	if(write_en & (address == 6'h1A))
		reg_file[1A]     <=  wr_data[7:0];
end

assign brake3       = reg_file[1A][7];
assign enable3      = reg_file[1A][6];
assign direction3   = reg_file[1A][5];
assign pwm3[4:0]    = reg_file[1A][4:0];

// ------------- 0x1B	ROTATION3_CURR_ANG	-------------
always @(posedge clock) begin
	if(write_en & (address == 6'h1B))
		reg_file[1B]     <=  wr_data[7:0];
end

assign brake3       = reg_file[1B][7];
assign enable3      = reg_file[1B][6];
assign direction3   = reg_file[1B][5];
assign pwm3[4:0]    = reg_file[1B][4:0];

