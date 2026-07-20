import numpy as np
import pandas as pd
import xarray as xr
from contrailstereo.data.caliop import build_scene_table


def test_scene_table_grouping_and_sort(tmp_path):
    n = 30
    ds = xr.Dataset({
        "caliop_time": ("n", np.array(["2020-06-01T10:00"] * n,
                                      dtype="datetime64[ns]")),
        "caliop_lat": ("n", np.linspace(35, 40, n)),
        "caliop_lon": ("n", np.full(n, -110.0)),
        "top_altitude": ("n", np.full(n, 11000.0)),
        "goes_filename": ("n", np.array(["fileA"] * 20 + ["fileB"] * 10)),
    })
    p = tmp_path / "colloc.nc"
    ds.to_netcdf(p)
    scenes, prof = build_scene_table(str(p))
    assert list(scenes.goes_file) == ["fileA", "fileB"]   # descending n
    assert scenes.iloc[0].n == 20 and len(prof) == 30
    assert abs(scenes.iloc[0].top_km - 11.0) < 1e-9


def test_crop_filters_apply(tmp_path):
    ds = xr.Dataset({
        "caliop_time": ("n", np.array(["2018-06-01", "2020-06-01"],
                                      dtype="datetime64[ns]")),
        "caliop_lat": ("n", np.array([37.0, 37.0])),
        "caliop_lon": ("n", np.array([-110.0, -110.0])),
        "top_altitude": ("n", np.array([11000.0, 11000.0])),
        "goes_filename": ("n", np.array(["fA", "fB"])),
    })
    p = tmp_path / "c.nc"
    ds.to_netcdf(p)
    scenes, prof = build_scene_table(str(p))
    assert len(prof) == 1 and prof.iloc[0].goes_file == "fB"   # 2018 dropped