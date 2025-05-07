import os
from pathlib import Path
from typing import Literal, Optional
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from tsl.data.datamodule import SpatioTemporalDataModule, TemporalSplitter
from tsl.data.loader import StaticGraphLoader
from tsl.data.preprocessing import StandardScaler
from tsl.engines import Predictor
from tsl.metrics.torch import MaskedMAE, MaskedMAPE, MaskedMSE, rmse

from src.gat import PFGAT
from src.gnn_dataset import ERA5Dataset
from src.gnn_dataset_multi import MultiERA5Dataset
from src.utils import get_adj_matrix


# dask.config.set(scheduler='synchronous')  # Do this in __getitem__
# from pytorch_lightning.loggers import TensorBoardLogger



loss_fn = MaskedMSE()

metrics = {
    "mae": MaskedMAE(),
    "mape": MaskedMAPE(),
    "mase": MaskedMSE(),
    # "rmse": rmse,
}

# setup predictor

model = PFGAT(
    hist_channels=1,
    cov_channels=38,
    hidden_channels=256,
    gat_heads=8,
    gat_out=64,
    horizon=1,
    encoder_version=2,
    encoder_layers=2,
)
class MyPredictor(Predictor):
    
    def log_loss(self, name, loss, **kwargs):
        """"""
        self.log(name + '_loss',
                 loss.detach(),
                 on_step=True,
                 on_epoch=True,
                 logger=True,
                 prog_bar=False,
                 **kwargs)
EPOCHS = 5
BATCHES_PER_EPOCH = 300



predictor = MyPredictor(
    model=model,
    optim_class=torch.optim.Adam,
    optim_kwargs={"lr": 0.001},
    loss_fn=loss_fn,
    metrics=metrics,
    scheduler_class=torch.optim.lr_scheduler.CosineAnnealingLR,
    scheduler_kwargs={
        "T_max": EPOCHS * BATCHES_PER_EPOCH,
        "eta_min": 0.0002,
    },
)
predictor.automatic_optimization = True 


checkpoint_callback = ModelCheckpoint(
    dirpath="logs",
    save_top_k=3,
    monitor="val_mse",
    mode="min",
)
# logger = TensorBoardLogger(
#     save_dir="tb_logs",
#     name="stgat_experiment"
# )

trainer = pl.Trainer(
    max_epochs=100,
    num_sanity_val_steps=0,
    limit_val_batches=24,
    #  logger=logger,
    #  gpus=0 if torch.cuda.is_available() else None,
    limit_train_batches=300, 
    callbacks=[checkpoint_callback],
    # precision="bf16" if torch.cuda.is_bf16_supported() else "16",
)

data_path = Path(__file__).parent / "data" / "geographic"
# Normalize data using mean and std computed over time and node dimensions
scalers = {"target": StandardScaler(axis=(0, 1)), 
        #    "ERA5":StandardScaler(axis=(0, 1))
           }
# Split data sequentially:
#   |------------ dataset -----------|
#   |--- train ---|- val -|-- test --|
splitter = TemporalSplitter(val_len=0.001, test_len=0.2)

adj_mat, target_mapped, fips2idx = get_adj_matrix()
dataset = MultiERA5Dataset(
    target_mapped.resample("1h").median(),
    covariates=None,
    connectivity=adj_mat,
    fips2idx=fips2idx,
    # weather_zarr_url="gs://gcp-public-data-arco-era5/co/single-level-reanalysis.zarr-v2",
    weather_zarr_url="/media/drive2/jaydenfassett/era5_subset_2022.zarr",
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
    batch_size=8,
    workers=12,
    prefetch_factor=3,
)

trainer.fit(predictor, datamodule=dm)
