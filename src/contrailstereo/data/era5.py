"""ERA5 winds for the in-scan advection correction (v3.2 design)"""

from __future__ import annotations

import logging

from pathlib import Path

import os
import numpy as np
import xarray as xr
import pandas as pd

from ..types import WindProfile

log = logging.getLogger(__name__)

G0 = 9.80665


def cache_path(cache_dir, when, lat, lon) -> Path:
    tag = f"{pd.Timestamp(when):%Y%m%d%H%M}_{lat:+.1f}_{lon:+.1f}"
    return Path(cache_dir) / f"era5_uvz_{tag.replace('.', 'p')}.nc"


def request_times(when):
    """Get the two hourly analyses bracketing `when`"""

    t0 = pd.Timestamp(when).floor("h")
    groups = {}
    for t in (t0, t0 + pd.Timedelta(hours=1)):
        groups.setdefault(t.normalize(), []).append(f"{t.hour:02d}:00")
    return list(groups.items())


def _download(fname, when, lat, lon, levels):
    import cdsapi
    try: 
        client = cdsapi.Client(quiet=True)
    except TypeError:
        client = cdsapi.Client()
    parts = []
    for i, (day, hours) in enumerate(request_times(when)):
        part = Path(f"{fname}.{i}.part")
        client.retrieve(
            "reanalysis-era5-pressure-levels",
            {"product_type": ["reanalysis"],
             "variable": ["u_component_of_wind", "v_component_of_wind",
                          "geopotential"],
             "year": [str(day.year)], "month": [f"{day.month:02d}"],
             "day": [f"{day.day:02d}"], "time": hours,
             "pressure_level": [str(int(p)) for p in levels],
             "data_format": "netcdf", "download_format": "unarchived",
             "area": [lat + 2, lon - 2, lat - 2, lon + 2]},
        ).download(str(part))
        parts.append(part)
    dss = [xr.open_dataset(p) for p in parts]
    tdim = _tdim(dss[0])
    xr.concat(dss, dim=tdim).sortby(tdim).load().to_netcdf(fname)
    for ds, p in zip(dss, parts):
        ds.close()
        p.unlink()


def _tdim(ds):
    return "valid_time" if "valid_time" in ds.dims else "time"
 
 
def _level_dim(ds):
    return "pressure_level" if "pressure_level" in ds.dims else "level"


def profile_from_file(fname, when, lat, lon, levels=None) -> WindProfile:
    """Interpolate a cached ERA5 file to (when, lat, lon)"""
    era = xr.open_dataset(fname)
    if levels is not None:
        have = {int(p) for p in np.atleast_1d(era[_level_dim(era)].values)}
        if have != {int(p) for p in levels}:
            raise ValueError(f"{fname} has levels {sorted(have)}, "
                             f"requested {sorted(int(p) for p in levels)}")
    prof = (era.interp(latitude=lat, longitude=lon)
               .interp({_tdim(era): np.datetime64(pd.Timestamp(when))}))
    h = prof.z.values / G0 / 1000.0
    u, v = prof.u.values, prof.v.values
    if not np.all(np.isfinite(np.r_[h, u, v])):
        raise ValueError("non-finite winds after interpolation "
                         "(time or location outside the fetched field?)")
    return WindProfile(h, u, v, source="era5")
 
 
def fetch_winds(when, lat, lon, levels, cache_dir):
    """ERA5 wind profile at (when, lat, lon)"""
    fname = cache_path(cache_dir, when, lat, lon)
    try:
        if not fname.exists():
            fname.parent.mkdir(parents=True, exist_ok=True)
            _download(fname, when, lat, lon, levels)
        return profile_from_file(fname, when, lat, lon, levels)
    except Exception as e:                           # noqa: BLE001
        log.warning("ERA5 unavailable for %s (%.2f, %.2f): %s: %s",
                    when, lat, lon, type(e).__name__, e)
        return None


