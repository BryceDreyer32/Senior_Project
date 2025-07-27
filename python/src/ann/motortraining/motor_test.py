import torch
import numpy as np
import joblib
import random
#from model import MotorNet  # import your trained model class
from motor_training import Motor_ANN  # import the ANN class defined in motor_training.py

# Load model
#model = Motor_ANN()
#model.load_state_dict(torch.load('python/src/ann/motortraining/pwm_model.pt'))
#model.eval()

model = torch.jit.load("python/src/ann/motortraining/pwm_model.pt")
model.eval()

# Load scalers
input_scaler = joblib.load('python/src/ann/motortraining/input_scaler.pkl')
output_scaler = joblib.load('python/src/ann/motortraining/output_scaler.pkl')

with open("results.txt", "w") as file:

    for runs in range(1000):
        random_start = random.randint(0, 4095)
        random_delta = random.randint(5, 2000)
        random_end = (random_start + random_delta) % 4096

        # Input: Start_Angle, End_Angle, Angle_Change, Voltage
    #    raw_input = np.array([[1681,1845,164,12.5]])  # Example input
        raw_input = np.array([[random_start, random_end, random_delta,12.5]])  # Example input

        # Normalize input
        input_scaled = input_scaler.transform(raw_input)

        # Convert to tensor and predict
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
        with torch.no_grad():
            output_scaled = model(input_tensor)
            output_scaled = output_scaled.numpy()

        # Inverse transform output to get actual PWM and duration
        predicted_output = output_scaler.inverse_transform(output_scaled)
        
        file.write(f"Start = {random_start}, End = {random_end}, Delta = {random_delta}, Predicted PWM: {predicted_output[0][0]:.2f}, Duration: {predicted_output[0][1]:.4f}\n")

file.close()