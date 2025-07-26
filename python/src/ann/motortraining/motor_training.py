import torch
import torch.nn as nn
import numpy as np
import time
import random
import csv
import sys
import os

print(os.getcwd())
sys.path.append(os.path.realpath('python/src/constants'))
sys.path.append(os.path.realpath('python/src/subsystem/fpga'))
import Constants
import FpgaCommunication

# FPGA instance
fpga = FpgaCommunication.FpgaCommunication(Constants.Constants.FPGA_SPI_CHANNEL, Constants.Constants.FPGA_SPI_DEVICE, Constants.Constants.FPGA_SPI_MODE, Constants.Constants.FPGA_SPI_SPEED)


# --- ANN Model ---
class MotorANN(nn.Module):

    def __init__(self, motor):
        super().__init__()
        self.motor = motor
        self.net = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Tanh()  # Output: PWM in range [-1, 1]
        )
def forward(self, x):
        return self.net(x)


def read_encoder(self, motor):
    if(motor == 0):
        fpga.fpgaWrite(Constants.Constants.ROTATION0_CURRENT_ANGLE2_ADDR, 0x40)
        lsb_data = fpga.fpgaRead(Constants.Constants.ROTATION0_CURRENT_ANGLE_ADDR)
        msb_data = fpga.fpgaRead(Constants.Constants.ROTATION0_CURRENT_ANGLE2_ADDR)

    elif(motor == 1):
        fpga.fpgaWrite(Constants.Constants.ROTATION1_CURRENT_ANGLE2_ADDR, 0x40)
        lsb_data = fpga.fpgaRead(Constants.Constants.ROTATION1_CURRENT_ANGLE_ADDR)
        msb_data = fpga.fpgaRead(Constants.Constants.ROTATION1_CURRENT_ANGLE2_ADDR)

    elif(motor == 2):
        fpga.fpgaWrite(Constants.Constants.ROTATION2_CURRENT_ANGLE2_ADDR, 0x40)
        lsb_data = fpga.fpgaRead(Constants.Constants.ROTATION2_CURRENT_ANGLE_ADDR)
        msb_data = fpga.fpgaRead(Constants.Constants.ROTATION2_CURRENT_ANGLE2_ADDR)

    elif(motor == 3):
        fpga.fpgaWrite(Constants.Constants.ROTATION3_CURRENT_ANGLE2_ADDR, 0x40)
        lsb_data = fpga.fpgaRead(Constants.Constants.ROTATION3_CURRENT_ANGLE_ADDR)
        msb_data = fpga.fpgaRead(Constants.Constants.ROTATION3_CURRENT_ANGLE2_ADDR)

    rd_data = ((msb_data << 8) | lsb_data) & 0xFFF
    return rd_data

def apply_pwm(motor, pwm_val):
    print(f"Motor {motor} PWM Command: {pwm_val:.2f}")
    val = ((1 << 7) | (pwm_val & 0x3F))
    if(motor == 0):
        fpga.fpgaWrite(Constants.Constants.ROTATION0_CONTROL_ADDR, val)

    elif(motor == 1):
        fpga.fpgaWrite(Constants.Constants.ROTATION1_CONTROL_ADDR, val)

    elif(motor == 2):
        fpga.fpgaWrite(Constants.Constants.ROTATION2_CONTROL_ADDR, val)

    elif(motor == 3):
        fpga.fpgaWrite(Constants.Constants.ROTATION3_CONTROL_ADDR, val)


def normalize_angle(angle):
    return (angle / 4096.0) * 2 - 1  # Scale to [-1, 1]
    
# --- Trial Runner ---
class MotorTrainer:
    def __init__(self, motor):
        self.model = MotorANN(motor)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        self.prev_angle = read_encoder()
        self.prev_pwm = 0.0
        self.csvfile = open('motor_trial_data.csv', 'w', newline='')
        self.writer = csv.writer(self.csvfile)
        self.writer.writerow(['input1', 'input2', 'input3', 'input4', 'target_angle'])
    
    def run_trial(self, steps=100):
        for _ in range(steps):
            current_angle = read_encoder()
            angle_delta = current_angle - self.prev_angle
            if angle_delta > 2048:
                angle_delta -= 4096
            elif angle_delta < -2048:
                angle_delta += 4096
            angle_delta_norm = angle_delta / 2048.0  # Normalize to [-1, 1]

            # Random small angle target (20 deg ~ 227 encoder ticks)
            target_angle = current_angle + random.uniform(-227, 227)
            target_angle = target_angle % 4096
            error = target_angle - current_angle
            if error > 2048:
                error -= 4096
            elif error < -2048:
                error += 4096
            error_norm = error / 2048.0  # Normalize

            # Prepare NN input
            input_vec = np.array([
                normalize_angle(current_angle),
                angle_delta_norm,
                error_norm,
                self.prev_pwm
            ], dtype=np.float32)
            input_tensor = torch.tensor(input_vec)

            # Forward pass
            pwm_out = self.model(input_tensor).item()
            pwm_scaled = pwm_out * 1.0  # Scale to PWM range [-1, 1]

            # Apply PWM
            apply_pwm(pwm_scaled)

            # Log data
            self.writer.writerow(list(input_vec) + [normalize_angle(target_angle)])

            # Train immediately (online learning)
            target_norm = normalize_angle(target_angle)
            predicted_pos = normalize_angle(current_angle + pwm_scaled * 10.0)  # crude forward model
            loss = self.loss_fn(torch.tensor([predicted_pos]), torch.tensor([target_norm]))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update previous state
            self.prev_angle = current_angle
            self.prev_pwm = pwm_scaled

            time.sleep(0.01)  # 100 Hz loop

def close(self):
        self.csvfile.close()

if __name__ == "__main__":
    motor = 2
    trainer = MotorTrainer(motor)

    try:
        trainer.run_trial(steps=1000)
        
    finally:
        trainer.close()
