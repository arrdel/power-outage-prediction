"""
Author:
    Jack Morris

Description:
    Collection of functions for labeling which states a powerline starts and
    ends. These functions use Dask to speed up compute due to the O(nlog(d)+d)
    where n is the number of lat/lon points and d is number of districts.
"""

import copy
import datetime
import multiprocessing as mp
import pathlib

import dask.dataframe as dd
import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
from dask.distributed import Client, LocalCluster
from shapely.geometry import Point
from shapely.strtree import STRtree


def load_sectors():
    root_dir = pathlib.Path(__file__).parents[2] / "data" / "nypd"
    try:
        # data from https://data.cityofnewyork.us/Public-Safety/NYPD-Sectors/eizi-ujye
        sectors = joblib.load(root_dir / "sectors.pkl")
    except FileNotFoundError:
        # convert geometry column to shapely objects

        sectors = pd.read_csv(root_dir / "NYPD_Sectors_20240306.csv").convert_dtypes(
            dtype_backend="pyarrow"
        )
        sectors["the_geom"] = gpd.GeoSeries.from_wkt(sectors["the_geom"])
        sectors = gpd.GeoDataFrame(sectors, geometry="the_geom")

        joblib.dump(sectors, root_dir / "sectors.pkl")
    return sectors


def load_geography(filename: str):
    root_dir = pathlib.Path(__file__).parents[2] / "data" / "geographic"
    gdf = gpd.read_file(root_dir / filename)
    return gdf


def load_states():
    try:
        return load_geography("cb_2018_us_state_5m.shp")
    except FileNotFoundError:
        print(
            "Download shape file for states (5m) from: https://www.census.gov/geographies/mapping-files/time-series/geo/carto-boundary-file.html"
        )
        raise


def load_counties():
    try:
        return load_geography("cb_2018_us_county_500k.shp")
    except FileNotFoundError:
        print(
            "Download shape file for counties (500k) from: https://www.census.gov/geographies/mapping-files/time-series/geo/carto-boundary-file.html"
        )
        raise


def get_district(
    latitude: pa.float64, longitude: pa.float64, snames: np.ndarray, tree: STRtree
) -> str:
    """Fast function for finding which district a point lies inside

    Args:
        latitude (pa.float64): floating point latitude value
        longitude (pa.float64): floating point longitude value
        snames (np.ndarray): array of sector names indexed the same as the tree
        tree (STRtree): tree of geometries for quick lookup

    Returns:
        str: csv of sector names a point falls into
    """
    p = Point(longitude, latitude)
    idx = tree.query(p, "within")
    if len(idx):
        return snames[idx[0]]  # type:str
    return ""


def id_sector_precinct(
    df,
    states: gpd.GeoDataFrame | None = None,
    date_range: tuple[int, int] | None = None,
    date_col_name: str | None = "created_date",
):
    if states is None:
        states = load_states()
    id_state(df, states, inplace=True)
    id_county(df, "sector", date_range, date_col_name, inplace=True)


def id_state(
    df: pd.DataFrame,
    states: gpd.GeoDataFrame | None = None,
    # date_range:tuple[int,int]|None = None,
    # date_col_name:str|None = 'created_date',
    inplace: bool = False,
):
    if states is None:
        states = load_states()

    if not inplace:
        df = copy.deepcopy(df)
    assert not df[["latitude", "longitude"]].isna().sum().sum()
    df.dropna(how="any", subset=["latitude", "longitude"], inplace=True)

    # if date_range:
    #     df = df[(df[date_col_name]>= datetime.date(date_range[0], 1, 1))&(df[date_col_name] < datetime.date(date_range[1], 1, 1))]
    # Create snames list for use in labeling points
    snames: np.ndarray = states["STUSPS"].values
    # Build a tree of geometry for faster queries
    tree: STRtree = STRtree(states["geometry"].values)

    sct = _distributed_district_compute(df, snames, tree, "state")

    if inplace:
        # add computed values to dataframe
        df["state"] = sct
        print("Completed sector computation!")
    else:
        out = pd.Series(sct)
        print("Completed sector computation!")
        return out


def id_county(
    df: pd.DataFrame,
    county: gpd.GeoDataFrame | None = None,
    states: str | None = None,
    date_range: tuple[int, int] | None = None,
    date_col_name: str | None = "created_date",
    inplace: bool = False,
):
    if date_range:
        adf: pd.DataFrame = df[  # type:ignore
            (df[date_col_name] >= datetime.date(date_range[0], 1, 1))
            & (df[date_col_name] < datetime.date(date_range[1], 1, 1))
        ]
    else:
        adf = df

    if isinstance(states, str):
        county = strip_sector_to_precinct(adf[states])
    else:
        if county is None:
            county = load_precincts()
        # Create snames list for use in labeling points
        snames: np.ndarray = county["precinct"].values
        tree: STRtree = STRtree(county.geometry.values)
        county = _distributed_district_compute(adf, snames, tree, "precinct")

    if inplace:
        df["precinct"] = county
        print("Completed precinct computation!")

    else:
        print("Completed precinct computation!")
        return county


def strip_sector_to_precinct(sector: pd.Series) -> gpd.GeoDataFrame:
    return sector.str.slice(0, 3).str.lstrip("0").str.rstrip(r"ABCDEFGHIJKL")


def _distributed_district_compute(
    df: pd.DataFrame, snames: np.ndarray, tree: STRtree, col_name: str
):
    worker_count = int(0.9 * mp.cpu_count())  # leave a core or so for other use
    with (
        LocalCluster(
            n_workers=worker_count,
            processes=True,
            threads_per_worker=1,
            memory_limit="1GB",  # per worker memory limit
        ) as cluster,
        Client(cluster) as client,
    ):
        print("View progress here", client.dashboard_link)

        # convert to dask frame with worker_count partitions
        ddf: dd.DataFrame = dd.from_pandas(
            df[["latitude", "longitude"]], npartitions=worker_count
        )
        # apply district finder to all rows
        sct = ddf.apply(
            lambda x: get_district(x["latitude"], x["longitude"], snames, tree),
            axis=1,
            meta=(col_name, str),
        ).compute()
    return sct


def _distributed_district_compute_broken(df, snames, tree, col_name):
    worker_count = int(0.9 * mp.cpu_count())  # leave a core or so for other use
    with (
        LocalCluster(
            n_workers=worker_count,
            processes=True,
            threads_per_worker=1,
            memory_limit="1GB",  # per worker memory limit
        ) as cluster,
        Client(cluster) as client,
    ):
        print("View progress here", client.dashboard_link)

        # convert to dask frame with worker_count partitions
        ddf: dd.DataFrame = dd.from_pandas(
            df[["latitude", "longitude"]], npartitions=worker_count
        )
        # apply district finder to all rows
        sct = ddf.map_partitions(
            lambda partition: partition.apply(
                lambda x: get_district(x["latitude"], x["longitude"], snames, tree),
                axis=1,
            ),
            return_type=pd.Series([], dtype="string[pyarrow]"),
        )
    return sct.compute()
