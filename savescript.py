import xarray as xr

import scipy.spatial
import numpy as np
import geopandas as gpd 
import os
from pathlib import Path
def lon_to_360(dlon: float) -> float:
  return ((360 + (dlon % 360)) % 360)

ds = xr.open_zarr(
    'gs://gcp-public-data-arco-era5/co/single-level-reanalysis.zarr-v2',
    # chunks=None,
    # chunks={'time': 256},
    storage_options=dict(token='anon'),
    consolidated=True,
)

subset = ds.sel(time=slice("2022-01-01", "2022-12-31"))
# subset = ds.sel(time=slice("2022-01-01", "2022-01-03"))

subset = subset.assign_coords({
    'longitude': subset['longitude'].compute(),
    'latitude': subset['latitude'].compute()
})
# Now filter using where()
subset = subset.where(
    (subset.longitude >= lon_to_360(-171.79)) &
    (subset.longitude <= lon_to_360(-66.96)) &
    (subset.latitude >= 18.91) &
    (subset.latitude <= 71.35),
    drop=True
)
# result = subset.compute()

subset = subset.chunk({'time': 30})
subset = subset.reset_encoding()
subset.to_zarr("/media/drive2/jaydenfassett/era5_subset_2022.zarr", mode="w", compute=True)
## SAVE RESULT
# result.to_zarr("/media/drive2/jaydenfassett/era5_subset_2022.zarr", mode="w")
# result.to_zarr("/media/drive2/jaydenfassett/era5_subset_2022.zarr", mode="w")

