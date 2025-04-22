import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from pathlib import Path
from tsl.data import SpatioTemporalDataset
from src.GCN_hc import SpatioTemporalOutageModel  # your model
class OutageDataset(torch.utils.data.Dataset):
    def __init__(self, tensor, edge_index=None):
        self.x = tensor
        self.edge_index = edge_index

    def __getitem__(self, idx):
        # Example: input is a time step t and its graph
        x_t = self.x[idx]       # [N, F]
        y_t = self.x[idx]       # or some label derived from x_t
        return x_t, y_t, self.edge_index

    def __len__(self):
        return self.x.shape[0]
def main():
    # device & hyperparams
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 50
    batch_size = 32
    n_splits   = 5
    lr         = 1e-3

    # 1) load the full dataset
    tsl_path = Path(__file__).parent / "data" / "dataset.tsl"
    full_ds = torch.load(tsl_path, weights_only=False)
    assert isinstance(full_ds, SpatioTemporalDataset)
    _, _, F = full_ds.get_tensor("targets")[0].shape   # F = 14
    EMBED_DIM = 64  # or whatever
    LSTM_HID  = 128
    # 2) Get the core tensor
    tensor, _ = full_ds.get_tensor("target")  # Unpack the first item
    tensor = tensor.float().to(device)

    # Optional: Sanity check
    print(f"Loaded tensor shape: {tensor.shape}")
    # sanity check
    T, N, F = full_ds.shape
    print(f"Loaded spatio‐temporal data with T={T}, N={N}, F={F}")

    # prepare KFold splitter
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_rmse = []

    edge_index  = full_ds.edge_index
    edge_weight = full_ds.edge_weight

    graph_dataset = OutageDataset(tensor, edge_index=full_ds.edge_index)

    # 2) loop over folds

    for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(graph_dataset))), start=1):
        print(f"\n=== Fold {fold}/{n_splits} ===")
        # create train/val subsets
        train_ds = Subset(graph_dataset, train_idx)
        val_ds   = Subset(graph_dataset, val_idx)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                num_workers=4, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                                num_workers=4, pin_memory=True)
        

# Later you’ll use:
# model = SpatioTemporalOutageModel(edge_index=full_ds.edge_index, edge_weight=ew)

        # 3) fresh model, optimizer, loss
        model   = SpatioTemporalOutageModel(edge_index=edge_index,edge_weight=edge_weight,     feat_dim=F,
    embed_dim=EMBED_DIM,
    lstm_hid=LSTM_HID,
    num_nodes=N).to(device)
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
                y_pred = model(x)
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

if __name__ == "__main__":
    main()