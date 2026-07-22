import json
import os
import numpy as np
import pandas as pd
import pytest
import xarray as xr

FIXDIR = os.path.join(os.path.dirname(__file__), "fixtures")


def load_fixture_scene(scene_idx):
    from contrailstereo.core.retrieve import SceneData
    d = os.path.join(FIXDIR, f"scene_{scene_idx:04d}")
    meta = json.load(open(os.path.join(d, "meta.json")))
    fields = {sat: {ch: xr.open_dataset(
                  os.path.join(d, f"G{sat}_C{ch:02d}.nc"))
                  for ch in (13, 15)} for sat in (16, 17)}
    t16 = np.datetime64(fields[16][13].t.values)
    t17 = np.datetime64(fields[17][13].t.values)
    winds = None
    wp = os.path.join(d, "era5.npz")
    if os.path.exists(wp):
        w = np.load(wp)
        winds = (lambda hk: float(np.interp(hk, w["h"], w["u"])),
                 lambda hk: float(np.interp(hk, w["h"], w["v"])))
    return SceneData(
        when=pd.Timestamp(meta["time"]).floor("s").to_pydatetime(),
        lat0=meta["lat"], lon0=meta["lon"], fields=fields,
        domains={16: "C", 17: "C"},
        offset_s=float((t17 - t16) / np.timedelta64(1, "s")),
        profiles=pd.read_csv(os.path.join(d, "profiles.csv")).assign(
            goes_file=meta["goes_file"]),
        winds=winds, meta=meta)


@pytest.fixture
def fixture_scene():
    return load_fixture_scene