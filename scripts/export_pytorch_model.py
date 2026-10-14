"""
Export PyTorch Neural Network Weights to Vex Binary Format
===========================================================
This script trains or initializes a PyTorch model and exports its weights
into raw contiguous IEEE 754 64-byte aligned binary files (.bin) that can
be loaded directly into Vex tensors with zero-copy deserialization.
"""

import os
import torch
import torch.nn as nn
import numpy as np

# Deterministic seed for reproducible verification
torch.manual_seed(42)
np.random.seed(42)

batch_size = 8
in_features = 16
hidden1 = 32
hidden2 = 8
out_classes = 4

class SmallPretrainedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, out_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        z1 = self.fc1(x)
        a1 = self.relu(z1)
        z2 = self.fc2(a1)
        a2 = self.relu(z2)
        logits = self.fc3(a2)
        scaled = logits * 0.1
        return scaled

model = SmallPretrainedModel()
model.eval()

# Generate deterministic test input batch (8 samples x 16 features)
x_input = torch.randn(batch_size, in_features, dtype=torch.float32)

with torch.no_grad():
    pytorch_output = model(x_input)
    pytorch_sum = pytorch_output.sum().item()

print(f"PyTorch Output Shape : {tuple(pytorch_output.shape)}")
print(f"PyTorch Output Sum   : {pytorch_sum:.4f}")
print("First row of PyTorch logits:", pytorch_output[0].numpy())

# Ensure output directory exists
os.makedirs("models", exist_ok=True)

# PyTorch Linear weight is (out_features, in_features); transpose to (in_features, out_features) for X @ W
w1 = model.fc1.weight.detach().t().contiguous().numpy().astype(np.float32)
b1 = model.fc1.bias.detach().repeat(batch_size, 1).contiguous().numpy().astype(np.float32)

w2 = model.fc2.weight.detach().t().contiguous().numpy().astype(np.float32)
b2 = model.fc2.bias.detach().repeat(batch_size, 1).contiguous().numpy().astype(np.float32)

w3 = model.fc3.weight.detach().t().contiguous().numpy().astype(np.float32)
b3 = model.fc3.bias.detach().repeat(batch_size, 1).contiguous().numpy().astype(np.float32)

x_np = x_input.contiguous().numpy().astype(np.float32)

w1.tofile("models/w1.bin")
b1.tofile("models/b1.bin")
w2.tofile("models/w2.bin")
b2.tofile("models/b2.bin")
w3.tofile("models/w3.bin")
b3.tofile("models/b3.bin")
x_np.tofile("models/input.bin")

print(f"[OK] Exported 6 weight matrices and test input to 'models/' successfully.")
