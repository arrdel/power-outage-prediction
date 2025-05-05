from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from tsl.ops.connectivity import edge_index_to_adj

data_path = Path(__file__).parent.parent.parent / "data"


def get_adj_matrix():
    adj = pd.read_csv(data_path / "geographic/graph.csv")
    target = pd.read_csv(data_path / "eaglei_data/eaglei_outages_2022.csv")
    # sum all edges per county
    adj = adj.groupby(["src", "dest"])["total_voltage"].sum().reset_index()
    # get all unique counties
    unique = pd.unique(
        np.concat(
            (target["fips_code"].values.ravel(), adj[["src", "dest"]].values.ravel())
        )
    )
    fips2idx = pd.Series(index=unique, data=range(len(unique)))
    numeric_times = False
    target_mapped = target.copy()

    ### Map fips code to idx
    target_mapped["fips_code"] = target["fips_code"].map(fips2idx).astype(np.int64)
    ### Convert time to int

    target_mapped = target_mapped[["fips_code", "customers_out", "run_start_time"]]
    if numeric_times:
        target_mapped.loc[:, "slot"] = pd.to_datetime(
            target_mapped["run_start_time"]
        ).dt.floor("15min")

        unique_times = (
            target_mapped["slot"].drop_duplicates().sort_values().reset_index(drop=True)
        )

        slot2int = pd.Series(data=range(len(unique_times)), index=unique_times)
        target_mapped.loc[:, "run_start_time"] = target_mapped["slot"].map(slot2int)
        target_mapped["run_start_time"] = pd.to_numeric(
            target_mapped["run_start_time"], "raise"
        )
    else:
        target_mapped["run_start_time"] = pd.to_datetime(
            target_mapped["run_start_time"], "raise"
        )

    target_mapped = target_mapped[
        ["fips_code", "customers_out", "run_start_time"]
    ].pivot(index="run_start_time", columns="fips_code", values="customers_out")
    target_mapped.fillna(0, inplace=True)
    missing_cols = set(target_mapped.columns.to_numpy()).symmetric_difference(
        set(range(len(unique)))
    )
    new_cols = [target_mapped]
    for id in missing_cols:
        new_cols.append(
            pd.Series([0.0] * len(target_mapped), index=target_mapped.index, name=id)
        )
    target_mapped = pd.concat(new_cols, axis=1)
    adj_mapped = adj.copy()
    adj_mapped["src"] = adj["src"].map(fips2idx)
    adj_mapped["dest"] = adj["dest"].map(fips2idx)
    adj_mat = edge_index_to_adj(
        adj_mapped[["src", "dest"]].values.T,
        adj_mapped["total_voltage"].values,
        len(fips2idx),
    )
    return adj_mat, target_mapped


def get_weather_zarr():
    ds = xr.open_zarr(
        "gs://gcp-public-data-arco-era5/co/single-level-reanalysis.zarr-v2",
        # chunks=None,
        chunks={"time": 48},
        storage_options=dict(token="anon"),
        consolidated=True,
    )
    model_level_data = ds.sel(
        time=slice(ds.attrs["valid_time_start"], ds.attrs["valid_time_stop"])
    )
    return model_level_data
