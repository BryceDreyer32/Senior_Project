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


class MotorTrainer:
    def __init__(self, motor):
        self.model = MotorANN(motor)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

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
