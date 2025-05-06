from typing import Optional, Union, List, Tuple
import xarray as xr
import numpy as np
import scipy.spatial
import geopandas as gpd
from pathlib import Path
from tsl.data.spatiotemporal_dataset import SpatioTemporalDataset, parse_index
from tsl.data import (
    BatchMap,
    SpatioTemporalDataModule,
    StaticBatch,
    Data,
    BatchMapItem,
    SynchMode,
)
import warnings
from tsl.typing import DataArray, IndexSlice, SparseTensArray, TemporalIndex, TensArray


from tsl.data.preprocessing.scalers import Scaler, ScalerModule

from tsl.data.synch_mode import HORIZON, STATIC, WINDOW
import torch
import copy
# Suppress UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)
class Dummy:
    batch_pattern: str
    pattern: str
    synch_mode: SynchMode
    keys = []
    cat_dim = 0
    preprocess: bool
    shape = None


def lon_to_360(dlon: float) -> float:
    return (360 + (dlon % 360)) % 360


def roll_longitude(ds: xr.Dataset) -> xr.Dataset:
    """
    Shift longitude from [0, 360) to [-180, 180) ordering.
    """
    return ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180)).sortby(
        "longitude"
    )


def mirror_point_at_360(ds: xr.Dataset) -> xr.Dataset:
    """
    Duplicate the meridian (lon=0) at lon=360 to avoid wrap-around issues.
    """
    extra = ds.sel(longitude=0, method="nearest")
    extra = extra.assign_coords(longitude=extra.longitude + 360)
    return xr.concat([ds, extra], dim="longitude")



def build_triangulation(x, y):
  grid = np.stack([x, y], axis=1)
  return scipy.spatial.Delaunay(grid)


def interpolate(data, tri, mesh):
  indices = tri.find_simplex(mesh)
  ndim = tri.transform.shape[-1]
  T_inv = tri.transform[indices, :ndim, :]
  r = tri.transform[indices, ndim, :]
  c = np.einsum('...ij,...j', T_inv, mesh - r)
  c = np.concatenate([c, 1 - c.sum(axis=-1, keepdims=True)], axis=-1)
  result = np.einsum('...i,...i', data[:, tri.simplices[indices]], c)
  return np.where(indices == -1, np.nan, result)


class ERA5Dataset(SpatioTemporalDataset):
    """
    SpatioTemporalDataset for ERA5 weather data + power outage targets at county level.
    Data lives in GCS Zarr stores and is loaded lazily per time slice.
    """

    def __init__(
        self,
        target,
        covariates,
        connectivity,
        fips2idx,
        weather_zarr_url: str,
        county_shapefile: Path | str,
        window: int,
        horizon: int,
        delay: int = 0,
        stride: int = 1,
        storage_options: Optional[dict] = None,
    ):
        # initialize base sliding-window dataset (no in-memory target)
        super().__init__(
            target=target,
            covariates=covariates,
            connectivity=connectivity,
            window=window,
            horizon=horizon,
            delay=delay,
            stride=stride,
        )
        self.weather_zarr_url = weather_zarr_url
        self.storage_options = storage_options or {"token": "anon"}

        # load county geometries and centroids
        self.counties = gpd.read_file(county_shapefile)
        self.counties["centroid"] = self.counties.geometry.centroid
        self.counties["GEOID"] = self.counties["GEOID"].astype(fips2idx.index.dtype)
        county_mask = self.counties["GEOID"].isin(fips2idx.index)
        self.counties = self.counties[county_mask]
        # Rename fips2idx index to match GEOID
        fips2idx.index.name = "GEOID"
        # pre_reorder = copy.deepcopy(self.counties)
        # Reorder counties to match the order of fips2idx.index
        self.counties = self.counties.set_index("GEOID").loc[fips2idx.index].reset_index()

        # Verify the reordering
        assert all(self.counties["GEOID"].values == fips2idx.index.values), "Reordering failed: GEOID order mismatch"
        # for i, geoid in enumerate(fips2idx.index):
        #     assert self.counties.loc[self.counties["GEOID"] == geoid].equals(
        #     pre_reorder.loc[pre_reorder["GEOID"] == geoid]
        #     ), f"Row mismatch after {i} iterations for GEOID {geoid}, \n {self.counties.loc[self.counties["GEOID"] == geoid]} != \n\t {pre_reorder.loc[pre_reorder["GEOID"] == geoid]}"
        self.county_centroids = np.array(
            [[lon_to_360(pt.x), pt.y] for pt in self.counties.centroid]
        )

        # build triangulation over weather grid (using first timestep)
        ds = xr.open_zarr(
            self.weather_zarr_url,
            storage_options=self.storage_options,
            chunks={"time": window},  # type:ignore
            consolidated=True,
        )
        assert self.index is not None
        self.ds = ds.sel(time=slice(self.index.min().date(), self.index.max().date()))
        ds0 = self.ds.isel(time=slice(0,1)).compute()
        # print("selecting region")
        US_ds=ds0.where(
            (ds0.longitude > lon_to_360(-171.79)) & (ds0.latitude > 18.91) &
            (ds0.longitude < lon_to_360(-66.96)) & (ds0.latitude < 71.35),
            drop=True
        )
        # print("Buiding triangulation")
        self.grid_tri = build_triangulation(
            US_ds.longitude,
            US_ds.latitude,
        )
        # print("Triangulation done")
        era5_pattern = Dummy()
        era5_pattern.batch_pattern = "f t n"
        era5_pattern.pattern = "f t n"
        era5_pattern.preprocess = True
        era5_pattern.synch_mode = WINDOW

        self.input_map.__dict__["ERA5"] = era5_pattern
        self.patterns["ERA5"] = "f t n"

        # # store all available timestamps for indexing
        # ds_time = xr.open_zarr(
        #     self.outage_zarr_url, storage_options=self.storage_options
        # )
        # self.time_index = ds_time.time.values

    def add_weather_cov(
        self,
        name: str,
        value: DataArray,
        pattern: Optional[str] = None,
        add_to_input_map: bool = True,
        synch_mode: Optional[SynchMode] = None,
        preprocess: bool = True,
        convert_precision: bool = True,
    ):
        # validate name. name cannot be an attribute of self, but allow override
        self._check_name(name)
        value, pattern = self._parse_covariate(
            value, pattern, name=name, convert_precision=convert_precision
        )
        self._covariates[name] = dict(value=value, pattern=pattern)
        if add_to_input_map:
            self.input_map[name] = BatchMapItem(
                name,
                synch_mode,
                preprocess,
                cat_dim=None,
                pattern=pattern,
                shape=value.size(),
            )

    def get_time_slice(self, time_index):
        """
        Load weather covariates and outage targets for a given time range.
        Returns two xarray Datasets: covariates (vars × time × county) and targets.
        """
        # weather -> interpolate to counties

        ds_w = self.ds.isel(time=time_index).compute()
        cov = torch.tensor(self._interpolate(ds_w))
        

        return cov

    def _interpolate(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Interpolate all data_vars in ds from regular grid to county centroids.
        """
        tri = self.grid_tri
        centroid_coords = self.county_centroids
        # ds = ds.where(
        #     (ds.longitude > lon_to_360(-171.79)) & (ds.latitude > 18.91) &
        #     (ds.longitude < lon_to_360(-66.96)) & (ds.latitude < 71.35),
        #     drop=True
        # )
        interpolated_data = []
        for var in ds.data_vars:
            # print(f"Interpolating {var}")
            interpolated_data.append(interpolate(ds[var].values, tri, centroid_coords))
        return torch.tensor(interpolated_data)
 
    def _add_to_sample(
        self, out, synch_mode, endpoint="input", time_index=None, node_index=None
    ):
        # input_map
        batch_map: BatchMap = getattr(self, f"{endpoint}_map")
        for key, item in batch_map.by_synch_mode(synch_mode).items():
            if endpoint == "input" and key == "ERA5":
                weather_data = self.get_time_slice(time_index)
                if len(item.keys) > 1:
                    tensor, scaler = self.collate_weather_elem(
                        weather_data,
                        node_index=node_index,  # type:ignore
                        time_index=time_index,  # type:ignore
                    )
                else:
                    scaler = None
                    if key in self.scalers is not None:
                        scaler = self.scalers[key].slice(
                            time_index=time_index, node_index=node_index
                        )
                        tensor = scaler.transform(weather_data)
                    else:
                        tensor = weather_data

            else:
                if len(item.keys) > 1:
                    tensor, scaler = self.collate_item_elem(
                        key,
                        time_index=time_index,  # type:ignore
                        node_index=node_index,  # type:ignore
                    )
                else:
                    tensor, scaler = self.get_tensor(
                        item.keys[0],
                        preprocess=item.preprocess,
                        time_index=time_index,  # type:ignore
                        node_index=node_index,  # type:ignore
                    )
            if endpoint == "auxiliary":
                out[key] = tensor
            else:
                getattr(out, endpoint)[key] = tensor
            if scaler is not None:
                out.transform[key] = scaler

    def collate_weather_elem(
        self,
        ds,
        time_index: Union[List, torch.Tensor],
        node_index: Union[List, torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[ScalerModule]]:
        # expand and concatenate tensors
        x = ds
        # get scaler (if any)
        scaler = None
        if "ERA5" in self._batch_scalers:  # type:ignore
            scaler = self._batch_scalers["ERA5"].slice(  # type:ignore
                time_index=time_index, node_index=node_index
            )
            x = scaler.transform(x)
        return x, scaler


if __name__ == "__main__":
    from src.utils import get_adj_matrix
    from pathlib import Path

    data_path = Path(__file__).parent.parent / "data" / "geographic"

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
    print(dataset[0])