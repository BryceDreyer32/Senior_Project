# train_surrogate_nonlinear.py
import torch
import torch.nn as nn
import numpy as np

# Load dataset
d = np.load("python/src/mpc/mpc_dataset.npz")
X = torch.tensor(d['X'], dtype=torch.float32)
y = torch.tensor(d['y'], dtype=torch.float32).view(-1, 1)

# Train/test split
perm = torch.randperm(len(X))
train_idx = perm[:int(0.8*len(X))]
test_idx = perm[int(0.8*len(X)):]

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

# Simple MLP
class MPCSurrogate(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

model = MPCSurrogate()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# Train loop
for epoch in range(300):
    opt.zero_grad()
    pred = model(X_train)
    loss = loss_fn(pred, y_train)
    loss.backward()
    opt.step()

    if epoch % 50 == 0:
        test_loss = loss_fn(model(X_test), y_test).item()
        print(f"Epoch {epoch} TrainLoss={loss.item():.6f} TestLoss={test_loss:.6f}")

# Save PyTorch model
torch.save(model.state_dict(), "python/src/mpc/mpc_surrogate_nonlinear.pth")

# Export to ONNX for NPU deployment
dummy_input = torch.zeros(1, 3)
torch.onnx.export(model, dummy_input, "python/src/mpc/mpc_surrogate_nonlinear.onnx", input_names=['state'], output_names=['u'])
print("Saved ONNX model.")
