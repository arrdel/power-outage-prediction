import os
from pathlib import Path
from typing import Literal, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from tsl.data.datamodule import SpatioTemporalDataModule, TemporalSplitter
from tsl.data.loader import StaticGraphLoader
from tsl.data.preprocessing import StandardScaler
from tsl.engines import Predictor
from tsl.metrics.torch import MaskedMAE, MaskedMAPE

from src.gat import PFGAT
from src.gnn_dataset import ERA5Dataset
from src.utils import get_adj_matrix

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import dask
dask.config.set(scheduler='synchronous')  # Do this in __getitem__
# from pytorch_lightning.loggers import TensorBoardLogger
data_path = Path(__file__).parent / "data" / "geographic"
# Normalize data using mean and std computed over time and node dimensions
scalers = {"target": StandardScaler(axis=(0, 1))}
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
    precision="full",
)


class ERA5DataModule(SpatioTemporalDataModule):
    def __init__(self,
        prefetch_factor: int = 6,
        **kwargs,):
        super().__init__(**kwargs)
        self.prefetch_factor = prefetch_factor
        
    def get_dataloader(
        self,
        split: Literal["train", "val", "test"] = None,
        shuffle: bool = False,
        batch_size: Optional[int] = None,
    ) -> Optional[DataLoader]:
        if split is None:
            dataset = self.torch_dataset
        elif split in ["train", "val", "test"]:
            dataset = getattr(self, f"{split}set")
        else:
            raise ValueError(
                "Argument `split` must be one of 'train', 'val', or 'test'."
            )
        if dataset is None:
            return None
        pin_memory = self.pin_memory if split == "train" else None
        return StaticGraphLoader(
            dataset,
            batch_size=batch_size or self.batch_size,
            shuffle=shuffle,
            drop_last=split == "train",
            num_workers=self.workers,
            pin_memory=pin_memory,
            prefetch_factor=self.prefetch_factor,
        )


dm = ERA5DataModule(
    dataset=dataset,
    scalers=scalers,
    splitter=splitter,
    batch_size=2,
    workers=12
)


loss_fn = MaskedMAE()

metrics = {
    "mae": MaskedMAE(),
    "mape": MaskedMAPE(),
}

# setup predictor

model = PFGAT(
    hist_channels=1,
    cov_channels=38,
    hidden_channels=192,
    gat_heads=2,
    gat_out=64,
    horizon=1,
)

predictor = Predictor(
    model=model,
    optim_class=torch.optim.Adam,
    optim_kwargs={"lr": 0.001},
    loss_fn=loss_fn,
    metrics=metrics,
)


checkpoint_callback = ModelCheckpoint(
    dirpath="logs",
    save_top_k=1,
    monitor="val_mae",
    mode="min",
)
# logger = TensorBoardLogger(
#     save_dir="tb_logs",
#     name="stgat_experiment"
# )

trainer = pl.Trainer(
    max_epochs=100,
    num_sanity_val_steps=0,
    #  logger=logger,
    #  gpus=0 if torch.cuda.is_available() else None,
    limit_train_batches=100,  # end an epoch after 100 updates
    callbacks=[checkpoint_callback],
)

trainer.fit(predictor, datamodule=dm)
