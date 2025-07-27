import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import sklearn
print("version = " + sklearn.__version__)

# Load your dataset
df = pd.read_csv('python/src/ann/motortraining/motor_data.csv')

df['Current Angle'] = df['Start_Angle']
df['Target Angle'] = df['End_Angle']
df['Angle Delta'] = df['Angle_Change']
X = df[['Current Angle', 'Target Angle', 'Angle Delta', 'Voltage']]
y = df[['Speed', 'Duration']].values  # Replace 'Speed' if needed

# Fit scalers
input_scaler = StandardScaler().fit(X)
output_scaler = StandardScaler().fit(y)

# Normalize
X_scaled = input_scaler.transform(X)
y_scaled = output_scaler.transform(y)

#X = df[['Start_Angle', 'End_Angle', 'Voltage']].values
#y = df[['Speed', 'Duration']].values  # Replace 'Speed' if needed

# Normalize
x_scaler = StandardScaler()
y_scaler = StandardScaler()
X_scaled = x_scaler.fit_transform(X)
y_scaled = y_scaler.fit_transform(y)

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# DataLoaders
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=16)

def penalized_duration_loss(output, target, penalty_weight):
    mse = F.mse_loss(output, target)
    duration_penalty = output[:, 1].mean()  # assumes index 1 is duration
    return mse + penalty_weight * duration_penalty

# --- Define ANN ---
class Motor_ANN(nn.Module):
    def __init__(self):
        super(Motor_ANN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Dropout(0.1),  # 10% of neurons randomly disabled
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Dropout(0.1),  # 10% of neurons randomly disabled
            nn.Linear(16, 2)
        )

    def forward(self, x):
        return self.net(x)
    
if __name__ == '__main__':
    model = Motor_ANN()
    #criterion = nn.SmoothL1Loss()  # aka Huber loss
    # criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adjusted learning rate

    # --- Train ---
    train_losses = []
    epochs = 500

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for xb, yb in train_loader:
            pred = model(xb)
            #loss = criterion(pred, yb)

            loss = penalized_duration_loss(pred, yb, penalty_weight=0.1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

    # --- Evaluate ---
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_test).numpy()
        y_true_scaled = y_test.numpy()

        # Inverse transform to original units
        y_pred = y_scaler.inverse_transform(y_pred_scaled)
        y_true = y_scaler.inverse_transform(y_true_scaled)

    # --- Metrics ---
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f"\nEvaluation Metrics:")
    print(f"MAE : {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²  : {r2:.4f}")

    # --- Plot: Training Loss Curve ---
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Save the model ---
    example_input = torch.rand(1, 4)  # assuming input has 4 features
    scripted_model = torch.jit.trace(model, example_input)
    scripted_model.save("python/src/ann/motortraining/pwm_model.pt")

    # --- Save scalers ---
    import joblib

    # After fitting
    joblib.dump(input_scaler, 'python/src/ann/motortraining/input_scaler.pkl')
    joblib.dump(output_scaler, 'python/src/ann/motortraining/output_scaler.pkl')

    # --- Export to ONNX ---
    dummy_input = torch.randn(1, 4)  # assuming input has 4 features
    torch.onnx.export(
        model,  # your trained model
        dummy_input,
        "pwm_model.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=11
    )



# # --- Plot: Predicted vs Actual ---
# plt.figure(figsize=(6, 6))
# plt.scatter(y_true[:, 0], y_pred[:, 0], label='PWM', alpha=0.6)
# plt.scatter(y_true[:, 1], y_pred[:, 1], label='Duration', alpha=0.6)
# plt.plot([min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())],
#          [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())], 'r--')
# plt.xlabel("True Values")
# plt.ylabel("Predicted Values")
# plt.title("Predicted vs True (PWM & Duration)")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # --- Plot: Residuals ---
# residuals = y_pred - y_true
# plt.figure(figsize=(6, 4))
# plt.scatter(range(len(residuals)), residuals[:, 0], alpha=0.6, label='PWM Residuals')
# plt.scatter(range(len(residuals)), residuals[:, 1], alpha=0.6, label='Duration Residuals')
# plt.axhline(0, color='red', linestyle='--')
# plt.title("Residuals")
# plt.xlabel("Sample Index")
# plt.ylabel("Prediction Error")
# plt.legend()
# plt.tight_layout()
# plt.grid(True)
# plt.show()
