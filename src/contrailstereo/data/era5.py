"""ERA5 winds for the in-scan advection correction (v3.2 design)"""

from __future__ import annotations

import os
import numpy as np
import xarray as xr

from ..config import StereoConfig, DEFAULT


def fetch_era5_winds(when, lat0, lon0, cfg: StereoConfig = DEFAULT):
    os.makedirs(cfg.era5_cache, exist_ok=True)
    tag = f"{when:%Y%m%d%H%M}_{lat0:+.1f}_{lon0:+.1f}".replace(".", "p")
    fname = os.path.join(cfg.era5_cache, f"era5_uvz_{tag}.nc")
    try:
        if not os.path.exists(fname):
            import cdsapi                          # lazy
            try:
                client = cdsapi.Client(quiet=True)
            except TypeError:                      # older cdsapi
                client = cdsapi.Client()
            hours = [f"{when.hour:02d}:00",
                     f"{(when.hour + 1) % 24:02d}:00"]
            client.retrieve(
                "reanalysis-era5-pressure-levels",
                {"product_type": ["reanalysis"],
                 "variable": ["u_component_of_wind",
                              "v_component_of_wind", "geopotential"],
                 "year": [str(when.year)], "month": [f"{when.month:02d}"],
                 "day": [f"{when.day:02d}"], "time": hours,
                 "pressure_level": list(cfg.wind_levels),
                 "data_format": "netcdf", "download_format": "unarchived",
                 "area": [lat0 + 2, lon0 - 2, lat0 - 2, lon0 + 2]},
            ).download(fname)
        era = xr.open_dataset(fname)
        tdim = "valid_time" if "valid_time" in era.dims else "time"
        prof = era.interp(latitude=lat0, longitude=lon0) \
                  .interp({tdim: np.datetime64(when)})
        h = (prof.z.values / 9.80665) / 1000.0
        o = np.argsort(h)
        h, u, v = h[o], prof.u.values[o], prof.v.values[o]
        return (lambda hk: float(np.interp(hk, h, u)),
                lambda hk: float(np.interp(hk, h, v)))
    except Exception as e:                         # noqa: BLE001
        print(f"    ERA5 unavailable ({type(e).__name__}): "
              f"no wind correction")
        return None