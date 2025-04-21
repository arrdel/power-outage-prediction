import torch
import torch.nn as nn
from tsl.nn.blocks.encoders import SpatioTemporalTransformerLayer
from tsl.data import SpatioTemporalDataset
from torch_geometric.nn import GATConv
from pathlib import Path
import torch.nn.functional as F




class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, gat_heads):
        super().__init__()
        self.attn = SpatioTemporalTransformerLayer(
            input_size=in_channels,
            hidden_size=hidden_channels,
            ff_size=2*hidden_channels,
            activation='relu',
            dropout=0.1,
        )
        self.gat = GATConv(hidden_channels, hidden_channels, heads=gat_heads, concat=False)

    def forward(self, x, edge_index):
        x = x.permute(0, 2, 1, 3) # B F T N
        x = self.attn(x)              # → (B, N, hidden_channels)
        # B, N, T, H  = x.shape
        return x # B, N, T, H

class PFGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 gat_heads=4, output_features=14):
        super().__init__()
        # 1) Encoder
        self.encoder = Encoder(in_channels,hidden_channels,gat_heads)

        # 2) Two GATConv layers
        self.gat1 = GATConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels // gat_heads,
            heads=gat_heads,
            concat=True
        )
        self.gat2 = GATConv(
            in_channels=hidden_channels,
            out_channels=out_channels,
            heads=1,
            concat=False
        )
        # 3) Final FFN output
        self.ffn = nn.Linear(out_channels, output_features)


    def forward(self, x, edge_index):
        shape = x.shape
        assert len(shape) == 4, "add batch dim"
        B, T, N, n_features = shape
        x = self.encoder(x,edge_index) # B, N, T, H
        # exit()
        x = x[:,:,-1,:] # Taking the last step since it's been

        
        x = x.view(B * N, -1)         # → (B*N, hidden_channels)
        # replicate edges across batch
        edge_index = edge_index.repeat(1, B) + \
                     torch.arange(B, device=x.device).repeat_interleave(edge_index.size(1)) * N

        # graph attention
        x = F.elu(self.gat1(x, edge_index))
        x = self.gat2(x, edge_index)  # → (B*N, out_channels)

        # un-flatten and final MLP
        x = x.view(B, N, -1)          # → (B, N, out_channels)
        x = self.ffn(x)               # → (B, N, horizon)

        return x.permute(0, 2, 1)     # → (B, horizon, N)

# if __name__ == "__main__":
    # path = "/home/jaydenfassett/amlproject/data/county_data"
    # data = Path(__file__).parent.parent/"data"/"tsldataset"/"dataset.tsl"

    # # dataset = SpatioTemporalDataset.load(data)

    # dataset: SpatioTemporalDataset = torch.load(
    #     data,
    #     weights_only=False,)
    # print(dataset)



    # sample = dataset[0]
    # x = sample.x # T, N, F
    # x = x.unsqueeze(0)
    # y = sample.y

    # # print(x.shape)
    # # exit()
    # edge = sample.edge_index

    # qq = PFGAT(in_channels=14,hidden_channels=192,gat_heads=2,out_channels=64)


    # pred = qq.forward(x,edge)
    # print("y shape",y.shape)

    # print("output",pred.shape)