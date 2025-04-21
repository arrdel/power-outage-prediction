import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from pathlib import Path
from tsl.data import SpatioTemporalDataset
from src.GCN_hc import SpatioTemporalOutageModel  # your model

# device & hyperparams
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 50
batch_size = 32
n_splits   = 5
lr         = 1e-3

# 1) load the full dataset
tsl_path = Path(__file__).parent / "data" / "dataset.tsl"
full_ds  = torch.load(tsl_path)            # -> SpatioTemporalDataset
full_ds  = full_ds.float().to(device)      # float & move to device
assert isinstance(full_ds, SpatioTemporalDataset)

# sanity check
T, N, F = full_ds.shape
print(f"Loaded spatio‐temporal data with T={T}, N={N}, F={F}")

# prepare KFold splitter
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
fold_rmse = []

# 2) loop over folds
for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(full_ds))), start=1):
    print(f"\n=== Fold {fold}/{n_splits} ===")
    # create train/val subsets
    train_ds = Subset(full_ds, train_idx)
    val_ds   = Subset(full_ds, val_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    # 3) fresh model, optimizer, loss
    model   = SpatioTemporalOutageModel().to(device)
    opt     = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # 4) train/val loop
    for epoch in range(1, num_epochs+1):
        # — train
        model.train()
        train_loss = 0.0
        for x, y, edge in train_loader:
            x, y, edge = x.to(device), y.to(device), edge.to(device)
            opt.zero_grad()
            y_pred = model(x, edge)
            loss   = loss_fn(y_pred, y)
            loss.backward()
            opt.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_loader.dataset)

        # — validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y, edge in val_loader:
                x, y, edge = x.to(device), y.to(device), edge.to(device)
                y_pred = model(x, edge)
                val_loss += loss_fn(y_pred, y).item() * x.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch:02d}  Train RMSE: {train_loss**0.5:.4f}  "
              f"Val  RMSE: {val_loss**0.5:.4f}")

    fold_rmse.append(val_loss**0.5)

# 5) summary
avg_rmse = sum(fold_rmse) / n_splits
print(f"\n>>> Average validation RMSE over {n_splits} folds: {avg_rmse:.4f}")
