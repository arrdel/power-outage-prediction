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


#  Model Definition
class SpatioTemporalOutageModel(nn.Module):
    def __init__(self,
                 edge_index,
                 edge_weight=None,
                 feat_dim: int = 14,
                 embed_dim: int = 64,
                 lstm_hid: int  = 128,
                 num_nodes: int = 3233):
        super().__init__()
        self.edge_index  = edge_index
        self.edge_weight = edge_weight
        self.num_nodes   = num_nodes   
        # use the passed‑in dims, not hard‑coded constants:
        self.gcn1        = GCNConv(feat_dim, embed_dim)
        self.gcn2        = GCNConv(embed_dim, embed_dim)
        self.county_bias = nn.Embedding(num_nodes, embed_dim)
        self.lstm        = nn.LSTM(input_size=embed_dim,
                                   hidden_size=lstm_hid,
                                   batch_first=True)
        self.mlp         = nn.Sequential(
            nn.Linear(lstm_hid, lstm_hid // 2),
            nn.ReLU(),
            nn.Linear(lstm_hid // 2, 1),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor[T, N, F] = [time_steps, num_nodes, num_features]
        returns: Tensor[N] of outage logits or predictions
        """
        T, N, _ = x.shape
        assert N == self.num_nodes, f"Expected {self.num_nodes} nodes but got {N}"
        x = x.to(DEVICE)
        ei = self.edge_index
        ew = self.edge_weight
        
        # 1) Spatial encoding at each timestep
        embeds = []
        for t in range(T):
            h = self.gcn1(x[t], ei, ew)
            h = F.relu(h)
            h = self.gcn2(h, ei, ew)
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

