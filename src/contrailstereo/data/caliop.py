"""CALIOP-GOES collocations (Meijer 2024) as retrieval cases"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from ..geometry import bbox_around
from ..types import Case


def load_collocations(nc_path, lon_range=(-125.0, -95.0),
                      lat_range=(25.0, 50.0), min_year=2019) -> pd.DataFrame:
    """One row per collocated profile. 
    
    Columns: goes_file, time, lat, lon, top_km, index = pid"""

    ds = xr.open_dataset(nc_path)
    yr = ds.caliop_time.values.astype("datetime64[Y]").astype(int) + 1970
    lat, lon = ds.caliop_lat.values, ds.caliop_lon.values
    keep = np.where((yr >= min_year)
                    & (lon > lon_range[0]) & (lon < lon_range[1])
                    & (lat > lat_range[0]) & (lat < lat_range[1]))[0]
    sub = ds.isel(n=keep)

    out = pd.DataFrame({
        "goes_file": sub.goes_filename.values.astype(str),
        "time": pd.to_datetime(sub.caliop_time.values),
        "lat": sub.caliop_lat.values.astype(float),
        "lon": sub.caliop_lon.values.astype(float),
        "top_km": sub.top_altitude.values / 1000.0},
        index=pd.Index(keep, name="pid"))
    ds.close()
    return out


def scene_table(profiles: pd.DataFrame) -> pd.DataFrame:
    """Per-GOES-File summary"""

    t = (profiles.groupby("goes_file")
         .agg(n=("lat", "size"), time=("time", "median"),
              lat=("lat", "median"), lon=("lon", "median"),
              top_km=("top_km", "median"), top_sd=("top_km", "std"))
         .sort_values("n", ascending=False).reset_index())
    t.index.name = "scene"
    return t


def make_cases(profiles: pd.DataFrame, pad_lat=0.30, pad_lon=0.70):
    """One Case per GOES file in scene order.
 
    id    : GOES filename
    time  : median profile time, floored to the second (as before)
    bbox  : the track's extent plus padding (defaults reproduce the old grid)
    truth : that file's profiles (lat, lon, top_km, time), indexed by pid
    meta  : scene number, n, median top and its SD
    anchor: median profile position
    """
    groups = dict(tuple(profiles.groupby("goes_file")))
    cases = []
    for scene, row in scene_table(profiles).iterrows():
        pr = groups[row.goes_file]
        cases.append(Case(
            id=row.goes_file,
            time=pd.Timestamp(row.time).floor("s"),
            bbox=bbox_around(pr.lat, pr.lon, pad_lat, pad_lon),
            truth=pr[["lat", "lon", "top_km", "time"]].copy(),
            anchor=(float(row.lat), float(row.lon)),
            meta=dict(scene=int(scene), n=int(row.n),
                      top_km=float(row.top_km),
                      top_sd=float(row.top_sd) if pd.notna(row.top_sd)
                      else float("nan"))))
    return cases
 
 
def load_cases(nc_path, pad_lat=0.30, pad_lon=0.70, **filters):
    """load_collocations + make_cases."""
    return make_cases(load_collocations(nc_path, **filters), pad_lat, pad_lon)