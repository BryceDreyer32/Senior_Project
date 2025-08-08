import torch
import torch.nn as nn
import numpy as np
import random, math
import time, os, sys
import matplotlib.pyplot as plt
sys.path.append(os.path.realpath('python/src'))
sys.path.append(os.path.realpath('python/src/constants'))
sys.path.append(os.path.realpath('python/src/subsystem/fpga'))
import Constants
import FpgaCommunication
from hal.hal import HAL

# === Configuration ===
N_TRIALS = 5
ANGLE_MIN = 20    # degrees
ANGLE_MAX = 30   # degrees
DT = 0.02        # seconds per control step
TRIAL_DURATION = 2.0  # seconds
PWM_MIN = 5
PWM_MAX = 12

fpga = FpgaCommunication.FpgaCommunication(Constants.Constants.FPGA_SPI_CHANNEL, Constants.Constants.FPGA_SPI_DEVICE, Constants.Constants.FPGA_SPI_MODE, Constants.Constants.FPGA_SPI_SPEED)
hal = HAL(fpga)
motor = 2


class SurrogateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)
    
def set_pwm(motor, pwm_value, direction):
    if(pwm_value == 0):
        hal.stop_motor(motor)
    else:
        hal.run_motor(motor, pwm_value, direction)

def get_angle(motor): 
    return (hal.get_angle(motor)/ 4096.0) * 2 * 360.0

print('2 second wait...')
set_pwm(motor, 0, 1)
time.sleep(2)
print('Starting...')

# === Load trained surrogate model ===
model = SurrogateModel()
model.load_state_dict(torch.load("python/src/mpc/mpc_surrogate_nonlinear.pth"))
model.eval()

# === Data collection ===
all_errors = []

for trial in range(N_TRIALS):
    target_angle = random.uniform(ANGLE_MIN, ANGLE_MAX)
    print(f"\n--- Trial {trial+1} — Target: {target_angle:.2f} deg ---")

    trial_errors = []
    pwm = 0.0
    for t in np.arange(0, TRIAL_DURATION, DT):
        angle = get_angle(motor)
        error = target_angle - angle
        trial_errors.append(error)

        # --- Simple P controller for demo purposes ---
        pwm = 0.5 * error  # P gain
        pwm = int(np.clip(pwm, PWM_MIN, PWM_MAX))
        #print("Set PWM = " + str(pwm))
        set_pwm(motor, pwm, 1)

        time.sleep(DT)

    all_errors.append(trial_errors)

    # Stop motor between trials
    set_pwm(motor, 0, 1)
    time.sleep(1)

# === Plotting ===
plt.figure(figsize=(12, 6))
for i, trial_errors in enumerate(all_errors):
    plt.plot(np.arange(0, TRIAL_DURATION, DT), trial_errors, label=f"Trial {i+1}")

plt.xlabel("Time (s)")
plt.ylabel("Angle Error (deg)")
plt.title("Tracking Error Across Trials")
plt.grid(True)
plt.tight_layout()
plt.show()
