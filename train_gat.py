import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tsl.data.datamodule import (SpatioTemporalDataModule,
                                 TemporalSplitter)
from tsl.data.preprocessing import StandardScaler
from tsl.metrics.torch import MaskedMAE, MaskedMAPE
from tsl.engines import Predictor
import torch
from src.gnn_dataset import ERA5Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from src.gat import PFGAT

from src.utils import get_adj_matrix
from pathlib import Path
# from pytorch_lightning.loggers import TensorBoardLogger
data_path = Path("/home/jaydenfassett/powerup") / "data"
# Normalize data using mean and std computed over time and node dimensions
scalers = {'target': StandardScaler(axis=(0, 1))}

# Split data sequentially:
#   |------------ dataset -----------|
#   |--- train ---|- val -|-- test --|
splitter = TemporalSplitter(val_len=0.1, test_len=0.2)

adj_mat, target_mapped, fips2idx = get_adj_matrix()
dataset = ERA5Dataset(
    target_mapped.resample("1h").median(),
    covariates=None,
    connectivity=adj_mat,
    fips2idx=fips2idx,
    weather_zarr_url="gs://gcp-public-data-arco-era5/co/single-level-reanalysis.zarr-v2",
    county_shapefile=data_path / "cb_2018_us_county_500k.shp",
    window=12,
    horizon=1,
)

dm = SpatioTemporalDataModule(
    dataset=dataset,
    scalers=scalers,
    splitter=splitter,
    batch_size=1,
)




loss_fn = MaskedMAE()

metrics = {'mae': MaskedMAE(),
           'mape': MaskedMAPE(),
           'mae_at_15': MaskedMAE(at=2),  # '2' indicates the third time step,
                                          # which correspond to 15 minutes ahead
           'mae_at_30': MaskedMAE(at=5),
           'mae_at_60': MaskedMAE(at=11)}

# setup predictor

model = PFGAT(
    hist_channels=1,
    cov_channels=38,
    hidden_channels=192,
    gat_heads=2,
    gat_out=64,
    horizon=1,
)

class GATPredictor(Predictor):
    def training_step(self, batch, batch_idx):
        x_hist = batch.x.unsqueeze(-1)   # (B, T, N, 1)
        x_cov  = batch.ERA5               # (B, 38, T, N)
        y      = batch.y                 # (B, horizon, N)
        y_hat  = self.model(x_hist, x_cov, batch.edge_index)
        return super()._shared_step(y_hat, y, 'train')

    def validation_step(self, batch, batch_idx):
        x_hist = batch.x.unsqueeze(-1)
        x_cov  = batch.ERA5
        y      = batch.y
        y_hat  = self.model(x_hist, x_cov, batch.edge_index)
        return super()._shared_step(y_hat, y, 'val')

    def test_step(self, batch, batch_idx):
        x_hist = batch.x.unsqueeze(-1)
        x_cov  = batch.ERA5
        y      = batch.y
        y_hat  = self.model(x_hist, x_cov, batch.edge_index)
        return super()._shared_step(y_hat, y, 'test')

predictor = GATPredictor(
    model=model,
    optim_class=torch.optim.Adam,
    optim_kwargs={'lr': 0.001},
    loss_fn=loss_fn,
    metrics=metrics
)


checkpoint_callback = ModelCheckpoint(
    dirpath='logs',
    save_top_k=1,
    monitor='val_mae',
    mode='min',
)
# logger = TensorBoardLogger(
#     save_dir="tb_logs",
#     name="stgat_experiment"
# )
trainer = pl.Trainer(max_epochs=100,
                    #  logger=logger,
                    #  gpus=0 if torch.cuda.is_available() else None,
                     limit_train_batches=100,  # end an epoch after 100 updates
                     callbacks=[checkpoint_callback])

trainer.fit(predictor, datamodule=dm)