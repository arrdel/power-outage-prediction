'''
Huirong's try:

Choose Approach 1 (shared encoder + county bias) for the best trade off between interpretability and efficiency.

Start static (GCN → LSTM) to get a solid baseline quickly.

Use GCN+FFN initially; add attention (GAT, Graph Transformer) only if needed.

Evolve to Dynamic spatio-temporal GNNs as a second phase for maximum accuracy.
'''

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_self_loops

NUM_NODES   = 3143         
FEAT_DIM    = 10
EMBED_DIM   = 64
LSTM_HID    = 128
SEQ_LEN     = 12
LR          = 1e-3
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Build static graph topology from CSV
df = pd.read_csv('../data/geographic/graph.csv')
src = torch.tensor(df['src'].values, dtype=torch.long)
dst = torch.tensor(df['dest'].values, dtype=torch.long)
w   = torch.tensor(df['total_voltage'].values, dtype=torch.float)

# For an undirected graph, add reciprocal edges:
edge_index = torch.stack([src, dst], dim=0)
edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
edge_weight = torch.cat([w, w], dim=0)

edge_index  = edge_index.to(DEVICE)
edge_weight = edge_weight.to(DEVICE)

class OutagePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.gcn1 = GCNConv(FEAT_DIM, EMBED_DIM)
        self.gcn2 = GCNConv(EMBED_DIM, EMBED_DIM)
        self.county_bias = nn.Embedding(NUM_NODES, EMBED_DIM)
        self.lstm = nn.LSTM(input_size=EMBED_DIM, hidden_size=LSTM_HID, batch_first=True)
        self.mlp  = nn.Sequential(
            nn.Linear(LSTM_HID, LSTM_HID//2),
            nn.ReLU(),
            nn.Linear(LSTM_HID//2, 1)
        )
        
    def forward(self, weather_seq):
        T, N, _ = weather_seq.shape
        embeds = []
        for t in range(T):
            x = weather_seq[t]
            h = torch.relu(self.gcn1(x, edge_index, edge_weight))
            h = self.gcn2(h, edge_index, edge_weight)
            embeds.append(h)
        embeds = torch.stack(embeds, dim=0)
        bias   = self.county_bias(torch.arange(N, device=DEVICE))
        embeds = embeds + bias.unsqueeze(0)
        lstm_in   = embeds.permute(1, 0, 2)   # [N, T, D]
        out_seq, (h_n, _) = self.lstm(lstm_in)
        h_final   = h_n.squeeze(0)           # [N, LSTM_HID]
        preds     = self.mlp(h_final).squeeze(-1)
        return preds



model     = OutagePredictor().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.BCEWithLogitsLoss()



# need to build a real DataLoader that for each batch returns:
#   weather_seq: [T, NUM_NODES, FEAT_DIM]
#   targets:     [NUM_NODES] (0/1 outage or float severity)
# example:
def get_batch():
    # TODO: replace with real data loading
    weather_seq = torch.randn(SEQ_LEN, NUM_NODES, FEAT_DIM, device=DEVICE)
    targets      = torch.randint(0, 2, (NUM_NODES,), device=DEVICE).float()
    return weather_seq, targets

# Training Loop 
model.train()
for epoch in range(1, 51):
    weather_seq, targets = get_batch()
    optimizer.zero_grad()
    logits = model(weather_seq)
    loss = criterion(logits, targets)
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d}  Loss: {loss.item():.4f}")