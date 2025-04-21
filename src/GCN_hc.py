'''

Choose Approach 1 (shared encoder + county bias) for the best trade off between interpretability and efficiency.

Start static (GCN → LSTM) to get a solid baseline quickly.

Use GCN+FFN initially; add attention (GAT, Graph Transformer) only if needed.

'''


import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch_geometric.nn import GCNConv



DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load static graph from csv
df = pd.read_csv('./data/geographic/graph.csv')
src_zips  = df['src']
dst_zips  = df['dest']
voltages  = df['total_voltage']

# 2) build the ZIP→node‑id map
all_zips = pd.unique(pd.concat([src_zips, dst_zips]))
all_zips = list(all_zips)
id_map   = {z: i for i, z in enumerate(all_zips)}
NUM_NODES = len(all_zips)   # now dynamic!

# 3) map raw ZIPs → contiguous IDs
src = torch.tensor([id_map[z] for z in src_zips], dtype=torch.long)
dst = torch.tensor([id_map[z] for z in dst_zips], dtype=torch.long)
w   = torch.tensor(voltages.values, dtype=torch.float)

# 4) build undirected edges
edge_index  = torch.cat([
    torch.stack([src, dst], dim=0),
    torch.stack([dst, src], dim=0),
], dim=1)
edge_weight = torch.cat([w, w], dim=0)

# 5) move to device
edge_index  = edge_index.to(DEVICE)
edge_weight = edge_weight.to(DEVICE)


NUM_NODES    = len(all_zips)   # number of unique counties
FEAT_DIM     = 15     # your features per node (weather + outage, etc)
EMBED_DIM    = 64     # hidden dimension for GCN
LSTM_HID     = 128    # hidden dimension for LSTM


#  Model Definition
class SpatioTemporalOutageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 1) shared spatial encoder (2‐layer GCN)
        self.gcn1 = GCNConv(FEAT_DIM, EMBED_DIM)
        self.gcn2 = GCNConv(EMBED_DIM, EMBED_DIM)
        # 2) per‐county bias embeddings
        self.county_bias = nn.Embedding(NUM_NODES, EMBED_DIM)
        # 3) temporal decoder
        self.lstm = nn.LSTM(input_size=EMBED_DIM,
                            hidden_size=LSTM_HID,
                            batch_first=True)
        # 4) final MLP head
        self.mlp = nn.Sequential(
            nn.Linear(LSTM_HID, LSTM_HID // 2),
            nn.ReLU(),
            nn.Linear(LSTM_HID // 2, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor[T, N, F] = [time_steps, num_nodes, num_features]
        returns: Tensor[N] of outage logits or predictions
        """
        T, N, _ = x.shape
        assert N == NUM_NODES, f"..."
        x = x.to(DEVICE)
        
        # 1) Spatial encoding at each timestep
        embeds = []
        for t in range(T):
            h = self.gcn1(x[t], edge_index, edge_weight)
            h = F.relu(h)
            h = self.gcn2(h, edge_index, edge_weight)
            embeds.append(h)
        # embeds: list of T tensors [N, EMBED_DIM] → stack → [T, N, EMBED_DIM]
        embeds = torch.stack(embeds, dim=0)
        
        # 2) Add county‐bias: bias shape [N, EMBED_DIM] → broadcast over T
        bias   = self.county_bias(torch.arange(N, device=DEVICE))
        embeds = embeds + bias.unsqueeze(0)
        
        # 3) Prepare for LSTM: we want batch=N, seq_len=T
        #    so permute [T, N, D] → [N, T, D]
        seq = embeds.permute(1, 0, 2)
        
        # 4) Temporal decoding
        out_seq, (h_n, _) = self.lstm(seq)   # h_n: [1, N, LSTM_HID]
        h_final = h_n.squeeze(0)             # [N, LSTM_HID]
        
        # 5) Final prediction
        preds = self.mlp(h_final).squeeze(-1) # [N]
        return preds

if __name__ == "__main__":
    # dummy data: 37 time steps, 3233 counties, 15 features
    dummy = torch.randn(37, NUM_NODES, FEAT_DIM)
    
    model = SpatioTemporalOutageModel().to(DEVICE)
    logits = model(dummy)  # [3233]
    print("Output shape:", logits.shape)
