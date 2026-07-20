"""CALIOP collocation truth scene table from Meijer 2024"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from ..config import StereoConfig, DEFAULT


def build_scene_table(nc_path,
                      lon_range=(-125.0, -95.0),
                      lat_range=(25.0, 50.0),
                      min_year=2019):
    """Group collocated CALIOP profiles into per-GOES-file scenes.

    Scenes are sorted by profile count DESCENDING -- scene index is
    therefore a brightness/track-sampling ranking, not a neutral order
    (dev/held-out splits and tranche designs must account for this).

    Returns
    -------
    scenes : DataFrame, one row per GOES file
        n, time (median), lat/lon (median), top_km (median), top_sd.
    prof_df : DataFrame, one row per collocated profile
        goes_file, time, lat, lon, top_km.
    """
    ds = xr.open_dataset(nc_path)
    yr = ds.caliop_time.values.astype("datetime64[Y]").astype(int) + 1970
    keep = ((yr >= min_year)
            & (ds.caliop_lon.values > lon_range[0])
            & (ds.caliop_lon.values < lon_range[1])
            & (ds.caliop_lat.values > lat_range[0])
            & (ds.caliop_lat.values < lat_range[1]))
    sub = ds.isel(n=np.where(keep)[0])

    prof_df = pd.DataFrame({
        "goes_file": sub.goes_filename.values,
        "time": pd.to_datetime(sub.caliop_time.values),
        "lat": sub.caliop_lat.values,
        "lon": sub.caliop_lon.values,
        "top_km": sub.top_altitude.values / 1000.0})
    scenes = (prof_df.groupby("goes_file")
              .agg(n=("lat", "size"), time=("time", "median"),
                   lat=("lat", "median"), lon=("lon", "median"),
                   top_km=("top_km", "median"), top_sd=("top_km", "std"))
              .sort_values("n", ascending=False).reset_index())
    return scenes, prof_df