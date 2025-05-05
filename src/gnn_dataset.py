import xarray as xr
import numpy as np
import fsspec
import scipy.spatial
import geopandas as gpd
from pathlib import Path
from tsl.data.spatiotemporal_dataset import SpatioTemporalDataset


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


def build_triangulation(lons: np.ndarray, lats: np.ndarray) -> scipy.spatial.Delaunay:
    """
    Build a Delaunay triangulation over the 2D lon/lat grid.
    """
    # create mesh of points
    LON, LAT = np.meshgrid(lons, lats)
    points = np.vstack([LON.ravel(), LAT.ravel()]).T
    return scipy.spatial.Delaunay(points)


def interpolate(
    data: np.ndarray, tri: scipy.spatial.Delaunay, mesh: np.ndarray
) -> np.ndarray:
    """
    Barycentric interpolation of flattened data onto the mesh points.
    - data: array of shape (time, n_grid_points)
    - tri: Delaunay triangulation over grid points
    - mesh: array of shape (n_targets, 2) with lon/lat of target centroids
    Returns: interpolated array of shape (time, n_targets)
    """
    # find simplex for each target point
    simplex = tri.find_simplex(mesh)
    transform = tri.transform[simplex]
    coords = mesh - transform[:, 2]
    bary = np.einsum("ijk,ik->ij", transform[:, :2, :], coords)
    bary_coords = np.hstack([bary, 1 - bary.sum(axis=1, keepdims=True)])
    # gather values
    data_pts = data[:, tri.simplices[simplex]].transpose(
        1, 0, 2
    )  # (n_targets, time, 3)
    interp = np.einsum("ti,tri->tr", bary_coords, data_pts)
    interp[simplex == -1] = np.nan
    return interp


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
        weather_zarr_url: str,
        county_shapefile: str,
        window: int,
        horizon: int,
        delay: int = 0,
        stride: int = 1,
        storage_options: dict = None,
    ):
        # initialize base sliding-window dataset (no in-memory target)
        super().__init__(
            target=target,  # placeholder; we override __getitem__
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
        self.county_centroids = np.array(
            [[lon_to_360(pt.x), pt.y] for pt in self.counties.centroid]
        )

        # build triangulation over weather grid (using first timestep)
        ds = xr.open_zarr(
            self.weather_zarr_url,
            storage_options=self.storage_options,
            chunks={"time": 48},
            consolidated=True,
        )
        self.ds = ds
        ds0 = ds.isel(time=0).compute()
        # ds0 = roll_longitude(ds0)
        # ds0 = mirror_point_at_360(ds0)
        self.grid_tri = build_triangulation(
            np.unique(ds0.longitude.values),
            np.unique(ds0.latitude.values),
        )

        # # store all available timestamps for indexing
        # ds_time = xr.open_zarr(
        #     self.outage_zarr_url, storage_options=self.storage_options
        # )
        # self.time_index = ds_time.time.values

    def get_time_slice(self, start: str, end: str):
        """
        Load weather covariates and outage targets for a given time range.
        Returns two xarray Datasets: covariates (vars × time × county) and targets.
        """
        # weather -> interpolate to counties

        ds_w = self.ds.sel(time=slice(start, end)).compute()
        cov = self._interpolate(ds_w)

        return cov
    def _interpolate(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Interpolate all data_vars in ds from regular grid to county centroids.
        """
        # select US region if needed (optional)
        data = {}
        for var in ds.data_vars:
            arr = ds[var].values  # shape (time, lat, lon)
            nt, ny, nx = arr.shape
            flat = arr.reshape(nt, -1)
            interp = interpolate(flat, self.grid_tri, self.county_centroids)
            data[var] = (("time", "county"), interp)

        return xr.Dataset(
            data_vars=data,
            coords={
                "time": ds.time.values,
                "county": np.arange(self.county_centroids.shape[0]),
                "longitude": ("county", self.county_centroids[:, 0]),
                "latitude": ("county", self.county_centroids[:, 1]),
            },
        )

    def __getitem__(self, idx: int):
        # map sliding-window index to actual start/end timestamps
        start_idx, end_idx = self.indices[idx]
        start_time = np.datetime_as_string(self.time_index[start_idx-self.window])
        end_time = np.datetime_as_string(self.time_index[end_idx + self.horizon])
        cov, targ = self.get_time_slice(start_time, end_time)
        # format into tsl.Data via parent helper
        return super()._build_sample(
            target=targ.to_array().transpose("time", "county", ...),
            covariates=cov.to_array().transpose("time", "county", ...),
        )

    def __len__(self):
        return len(self.indices)  # number of sliding windows


if __name__ == "__main__":
    from src.utils import get_adj_matrix
    from pathlib import Path

    data_path = Path(__file__).parent.parent / "data" / "geographic"

    adj_mat, target_mapped = get_adj_matrix()
    ERA5Dataset(
        target_mapped.resample("1h").median(),
        covariates=None,
        connectivity=adj_mat,
        weather_zarr_url="gs://gcp-public-data-arco-era5/co/single-level-reanalysis.zarr-v2",
        county_shapefile=data_path / "cb_2018_us_county_500k.shp",
        window=12,
        horizon=1,
    )
