import torch
import numpy as np
import joblib
import random
import os, sys, time
#from model import MotorNet  # import your trained model class
from motor_training import Motor_ANN  # import the ANN class defined in motor_training.py

print(os.getcwd())
sys.path.append(os.path.realpath('python/src/constants'))
sys.path.append(os.path.realpath('python/src/subsystem/fpga'))
import Constants
import FpgaCommunication

# FPGA instance
fpga = FpgaCommunication.FpgaCommunication(Constants.Constants.FPGA_SPI_CHANNEL, Constants.Constants.FPGA_SPI_DEVICE, Constants.Constants.FPGA_SPI_MODE, Constants.Constants.FPGA_SPI_SPEED)

# Load model
#model = Motor_ANN()
#model.load_state_dict(torch.load('python/src/ann/motortraining/pwm_model.pt'))
#model.eval()

model = torch.jit.load("python/src/ann/motortraining/pwm_model.pt")
model.eval()

# Load scalers
input_scaler = joblib.load('python/src/ann/motortraining/input_scaler.pkl')
output_scaler = joblib.load('python/src/ann/motortraining/output_scaler.pkl')


def run_motor(motor, speed, duration):
    val = ((1 << 7) | (1 << 6) | (speed & 0x3F))
    if(motor == 0):
        fpga.fpgaWrite(Constants.Constants.ROTATION0_CONTROL_ADDR, val)
        time.sleep(duration)
        fpga.fpgaWrite(Constants.Constants.ROTATION0_CONTROL_ADDR, 0x0)

    elif(motor == 1):
        fpga.fpgaWrite(Constants.Constants.ROTATION1_CONTROL_ADDR, val)
        time.sleep(duration)
        fpga.fpgaWrite(Constants.Constants.ROTATION1_CONTROL_ADDR, 0x0)

    elif(motor == 2):
        fpga.fpgaWrite(Constants.Constants.ROTATION2_CONTROL_ADDR, val)
        time.sleep(duration)
        fpga.fpgaWrite(Constants.Constants.ROTATION2_CONTROL_ADDR, 0x0)

    elif(motor == 3):
        fpga.fpgaWrite(Constants.Constants.ROTATION3_CONTROL_ADDR, val)
        time.sleep(duration)
        fpga.fpgaWrite(Constants.Constants.ROTATION3_CONTROL_ADDR, 0x0)

def get_angle(motor):
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


with open("results.txt", "w") as file:

    try:
        for runs in range(10):
            print(f"Run {runs}\n")            
            start = get_angle(2)
            random_delta = random.randint(5, 2000)
            expected_end = (start + random_delta) % 4096

            print(f"Start = {start}, End = {expected_end}")

            # Input: Start_Angle, End_Angle, Angle_Change, Voltage
        #    raw_input = np.array([[1681,1845,164,12.5]])  # Example input
            raw_input = np.array([[start, expected_end, random_delta, 12.5]])  # Example input

            # Normalize input
            input_scaled = input_scaler.transform(raw_input)

            # Convert to tensor and predict
            input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
            with torch.no_grad():
                output_scaled = model(input_tensor)
                output_scaled = output_scaled.numpy()

            # Inverse transform output to get actual PWM and duration
            predicted_output = output_scaler.inverse_transform(output_scaled)
            
            pwm_val = int(predicted_output[0][0])
            if(pwm_val > 10):
                pwm_val = 10
            run_motor(2, pwm_val, float(predicted_output[0][1]))

            time.sleep(0.6)
            actual_end = get_angle(2)
            time.sleep(0.4)

            print(f", Actual End = {actual_end}, Error = {abs(actual_end-expected_end)}")

            file.write(f"Expected: Start = {start}, End = {expected_end}, Delta = {random_delta}, PWM: {pwm_val}, Duration: {predicted_output[0][1]:.4f}\n")
            file.write(f"Acual:    Start = {start}, End = {actual_end}, Delta = {actual_end-start}, Error = {abs(random_delta-(actual_end-start))}\n")
    except KeyboardInterrupt:
        fpga.fpgaWrite(Constants.Constants.ROTATION2_CONTROL_ADDR, 0x0)
        print("\nLoop interrupted by Ctrl+C.")
    finally:
        print("Exiting program.")


file.close()