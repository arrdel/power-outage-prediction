import torch
import torch.nn as nn
from tsl.nn.blocks.encoders import SpatioTemporalTransformerLayer
from tsl.data import SpatioTemporalDataset
from torch_geometric.nn import GATConv
from pathlib import Path
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops
from tsl.ops.connectivity import adj_to_edge_index
from tsl.nn.layers.base import NodeEmbedding
class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, gat_heads):
        super().__init__()
        self.attn = SpatioTemporalTransformerLayer(
            input_size=in_channels,
            hidden_size=hidden_channels,
            ff_size=2 * hidden_channels,
            activation="relu",
            dropout=0.1,
        )
        # we won’t use the inner GAT here, only the ST‐transformer

    def forward(self, x):
        # x: (B, T, N, C) → attn wants (B, N, T, C)
        x = x.permute(0, 2, 1, 3)
        return self.attn(x)  # → (B, N, hidden_channels)


class PFGAT(nn.Module):
    def __init__(
        self,
        hist_channels: int,
        cov_channels: int,
        hidden_channels: int,
        n_nodes: int = 3186,
        gat_heads: int = 4,
        gat_out: int = 64,
        horizon: int = 1,
    ):
        super().__init__()
        self.total_in = hist_channels + cov_channels
        self.emb = NodeEmbedding(n_nodes=n_nodes, emb_size=hidden_channels)

        self.encoder = Encoder(self.total_in, hidden_channels, gat_heads)

        # two GAT layers on the encoded node embeddings
        self.gat1 = GATConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels // gat_heads,
            heads=gat_heads,
            concat=True,
            # add_self_loops=False,

        )
        self.gat2 = GATConv(
            in_channels=hidden_channels,
            out_channels=gat_out,
            heads=1,
            concat=False,
            # add_self_loops=False,
        )

        # final linear to your forecast horizon
        self.ffn = nn.Linear(gat_out, horizon)

    def forward(self, x, ERA5, edge_index):
        """
        x_hist: (B, T, N, hist_channels) — your “input” field
        x_cov:  (B, f_cov, T, N)       — your ERA5 covariates
        edge_index: ([2, E])          — standard PyG edge index
        """
        x_hist = x                # (B, T, N, 1)
        x_cov  = ERA5             # (B, 38, T, N)

        B, T, N, _ = x_hist.shape
        edge_index = edge_index.to(torch.int32) # Converts to float & re-orders to get (2, E)
        edge_index = adj_to_edge_index(edge_index)[0]
        # 1) bring covariates into (B, T, N, cov_channels)
        # x_cov = x_cov.permute(0, 2, 3, 1)

        # 2) concatenate along feature‐axis → (B, T, N, total_in)
        x = torch.cat([x_hist, x_cov], dim=-1)
        

        # # 3) ST‐transformer → (B, N, hidden_channels)
        # x = self.encoder(x)

        # # 4) take the “last time” is already collapsed by the transformer,
        # #    so x is per‐node embedding
        # x = x.view(B * N, -1)

        # 5) replicate edge_index for batch
        # 3) ST‐transformer → (B, N, T, hidden_channels)
        # print("Are there nans in x?", torch.isnan(x).any())
        x_seq = self.encoder(x ) + self.emb()[None, :, None, :]
        # print("Are there nans in Encoder?", torch.isnan(x).any())
        # 4) take the last time step → (B, N, hidden_channels)
        x = x_seq[:, :, -1, :]
        # 5) flatten for GAT → (B*N, hidden_channels)
        x = x.view(B * N, -1)
        E = edge_index.size(1)
        
        edge_index = (
            edge_index.repeat(1, B)
            + torch.arange(B, device=x.device).repeat_interleave(E) * N
        )

        # 6) graph attention
        x = F.elu(self.gat1(x, edge_index))
        # print("Are there nans in GAT1?", torch.isnan(x).any())
        
        x = self.gat2(x, edge_index)  # → (B*N, gat_out)
        # print("Are there nans in GAT2?", torch.isnan(x).any())

        # 7) back to (B, N, gat_out), then per‐node horizon
        x = x.view(B, N, -1)
        
        x = self.ffn(x)  # → (B, N, horizon)

        # print("Are there nans FFN out?", torch.isnan(x).any())
        return x.unsqueeze(1)  # → (B, 1, N, horizon)


# if __name__ == "__main__":
#     from src.gnn_dataset import ERA5Dataset

#     path = "/home/jaydenfassett/amlproject/data/county_data"
#     data = Path(__file__).parent.parent / "data" / "tsldataset" / "dataset.tsl"

#     # dataset = SpatioTemporalDataset.load(data)

#     # dataset: SpatioTemporalDataset = torch.load(
#     #     data,
#     #     weights_only=False,)
#     # print(dataset)
#     from src.utils import get_adj_matrix
#     from pathlib import Path

#     data_path = Path("/home/jaydenfassett/powerup") / "data"

#     adj_mat, target_mapped = get_adj_matrix()
#     dataset = ERA5Dataset(
#         target_mapped.resample("1h").median(),
#         covariates=None,
#         connectivity=adj_mat,
#         weather_zarr_url="gs://gcp-public-data-arco-era5/co/single-level-reanalysis.zarr-v2",
#         county_shapefile=data_path / "cb_2018_us_county_500k.shp",
#         window=12,
#         horizon=1,
#     )
#     print(dataset)
#     print(dataset.shape)
#     sample = dataset[0]
#     x = sample.x  # T, N, F
#     x = x.unsqueeze(0)
#     y = sample.y

#     # print(x.shape)
#     # exit()
#     edge = sample.edge_index
#     qq = PFGAT(
#         in_channels=1,
#         hidden_channels=192,
#         gat_heads=2,
#         out_channels=64,
#         output_features=1,  # or whatever your horizon is
#     )

#     # qq = PFGAT(in_channels=14,hidden_channels=192,gat_heads=2,out_channels=64)

#     pred = qq.forward(x, edge)
#     print("y shape", y.shape)

#     print("output", pred.shape)
