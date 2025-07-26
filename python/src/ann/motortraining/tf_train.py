import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Load your dataset
df = pd.read_csv('D:\\Senior_Project_Local\\Senior_Project\\python\\src\\ann\\motortraining\\motor_data.csv')

# --- Feature Engineering ---
df['Current Angle'] = df['Start Angle']
df['Target Angle'] = df['End Angle']

# Input features (REMOVED 'Previous PWM')
X = df[['Current Angle', 'Target Angle', 'Voltage']].values

# Target labels
y = df[['Speed', 'Duration']].values  # Replace 'Speed' with PWM column if needed

# --- Normalize ---
x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_scaled = x_scaler.fit_transform(X)
y_scaled = y_scaler.fit_transform(y)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# DataLoaders
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=16)

# --- Define ANN (input size changed to 3) ---
class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # Output: PWM and Duration
        )

    def forward(self, x):
        return self.net(x)

model = ANN()

# Loss & optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# --- Train ---
for epoch in range(100):
    model.train()
    for xb, yb in train_loader:
        pred = model(xb)
        loss = criterion(pred, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

# --- Evaluate ---
model.eval()
with torch.no_grad():
    test_preds = model(X_test)
    test_loss = criterion(test_preds, y_test)
    print(f"\nTest Loss: {test_loss.item():.4f}")

# --- Example Inference ---
example_input = np.array([[20, 90, 12]])  # current angle, target angle, voltage
example_scaled = x_scaler.transform(example_input)
example_tensor = torch.tensor(example_scaled, dtype=torch.float32)

model.eval()
with torch.no_grad():
    pred_scaled = model(example_tensor)
    pred = y_scaler.inverse_transform(pred_scaled.numpy())
    print(f"\nPredicted PWM: {pred[0][0]:.2f}, Duration: {pred[0][1]:.2f} ms")

