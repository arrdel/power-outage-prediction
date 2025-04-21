import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from src.gat import STGAT
from pathlib import Path
from tsl.data import SpatioTemporalDataset
from tqdm import tqdm

def compute_rmse(y_pred, y_true):
    return torch.sqrt(torch.mean((y_pred - y_true) ** 2)).item()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = STGAT(in_channels=14,hidden_channels=128,out_channels=64).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
data = Path(__file__).parent/"data"/"tsldataset"/"dataset.tsl"

train_loader: SpatioTemporalDataset = torch.load(data,weights_only=False)

# === Training loop ===
num_epochs = 100
pbar = tqdm(range(num_epochs))
for epoch in pbar:
    model.train()
    for samp in train_loader:

        x, y = samp.x.to(device), samp.y.to(device)
        if len(x.shape) < 4:
            x = x.unsqueeze(0)
        edge = samp.edge_index.to(device)

        optimizer.zero_grad()
        y_pred = model(x,edge)

        loss = loss_fn(y_pred, y)

        loss.backward()
        optimizer.step()
    pbar.set_description(f"Epoch {epoch}, loss_step: {loss.item()}")

    # print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# # === Evaluation with RMSE ===
# model.eval()
# preds, targets = [], []
# with torch.no_grad():
#     for x, y in train_loader:
#         x, y = x.to(device), y.to(device)
#         y_pred = model(x)
#         preds.append(y_pred.cpu())
#         targets.append(y.cpu())

# preds = torch.cat(preds, dim=0)
# targets = torch.cat(targets, dim=0)
# rmse = compute_rmse(preds, targets)
# print(f"\nFinal RMSE: {rmse:.4f}")