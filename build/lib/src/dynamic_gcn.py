'''
Choose Approach 3 (county aware encoder): f(G_c) or f(G+C)
County-Specific Embeddings in GNN Message Passing
How it works:
    Learn a separate embedding vector per county (node) using an nn.Embedding.
    Concatenate these embeddings before or during message passing.
    This gives each node an identity vector, separate from its time-varying features.

Switch from GCN to a Dynamic ST-GNN
Instead of static edge weights:
    Dynamically compute edge weights or adjacency per timestep
    Use attention or similarity kernels to compute A_t
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch_geometric_temporal.nn.recurrent import GConvGRU

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# LOAD & PREPARE GRAPH TOPOLOGY 
df = pd.read_csv('./data/geographic/graph.csv')
src_zips = df['src']; dst_zips = df['dest']; voltages = df['total_voltage']

# Remap real ZIP codes to 0…N‑1 contiguous node indices
all_zips = pd.unique(pd.concat([src_zips, dst_zips]))
id_map   = {z: i for i, z in enumerate(all_zips)}
NUM_NODES = len(all_zips)

src = torch.tensor([id_map[z] for z in src_zips], dtype=torch.long)
dst = torch.tensor([id_map[z] for z in dst_zips], dtype=torch.long)
w   = torch.tensor(voltages.values, dtype=torch.float)

# Build undirected edges (stack forward + reverse) and move to device
edge_index  = torch.cat([
    torch.stack([src, dst], dim=0),
    torch.stack([dst, src], dim=0),
], dim=1).to(DEVICE)
edge_weight = torch.cat([w, w], dim=0).to(DEVICE)


# DYNAMIC SPATIO‑TEMPORAL MODEL
class DynamicSpatioTemporalOutageModel(nn.Module):
    def __init__(
        self,
        edge_index,
        edge_weight,
        num_nodes: int,
        feat_dim:  int = 15,
        embed_dim: int = 64,
        gru_K:     int = 2,
    ):
        super().__init__()
        self.edge_index  = edge_index
        self.edge_weight = edge_weight
        self.num_nodes   = num_nodes

        # CHANGE: 
        # from (embedding + county bias) to (county specific embeddings)
        # We learn a fixed embedding per county (node) to encode its identity.
        # This embedding is added into the message passing inputs, so the GNN
        # can modulate propagation based on county-specific biases.
        self.county_emb = nn.Embedding(num_nodes, embed_dim)

        # CHANGE: 
        # from static GCN+LSTM to Dynamic ST‑GNN (GConvGRU) 
        # At each time step, GConvGRU:
        #  - takes current node features + county embedding
        #  - aggregates neighbor messages (using graph conv)
        #  - updates hidden state via GRU recurrence
        # This tightly couples spatial & temporal modeling end‑to‑end.
        self.stggru = GConvGRU(
            in_channels  = feat_dim + embed_dim,  # dynamic + identity features
            out_channels = embed_dim,             # output embedding dim
            K            = gru_K,                 # polynomial filter order
        )

        # FINAL HEAD: simple MLP for prediction 
        # Takes the final hidden embedding per node and maps to a single output.
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [T, N, F] time-series of node features
        returns: [N] predicted outage severity or probability
        """
        T, N, F = x.shape
        assert N == self.num_nodes, f"Expected {self.num_nodes} nodes but got {N}"

        # Precompute the static county embedding vector for each node
        county_bias = self.county_emb(
            torch.arange(N, device=x.device)
        )  # shape: [N, embed_dim]

        h = None  # initial hidden state for GConvGRU

        #  CHANGE: Dynamic loop over time 
        # Instead of static GCN then LSTM, we loop T steps through our GConvGRU.
        for t in range(T):
            # 1) Concatenate dynamic features with county identity
            feat = torch.cat([x[t], county_bias], dim=-1)  # [N, F + embed_dim]

            # 2) One step of graph‑GRU: aggregates neighbors & updates state
            h = self.stggru(feat, self.edge_index, h)
            # h: [N, embed_dim]  hidden embedding at time t

        # After T steps, h holds the final spatio‑temporal embedding per node
        # Remain unchanged: MLP head 
        preds = self.mlp(h).squeeze(-1)  # [N]

        return preds


