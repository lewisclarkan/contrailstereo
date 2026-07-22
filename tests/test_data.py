import datetime as dt
import numpy as np
import pytest
import xarray as xr
from contrailstereo.config import StereoConfig
from contrailstereo.data.goes import (fetch_channel, row_artifact_scores,
                                      flag_rows, masked_copy)


def _fake_ds(ny=40, nx=60, seed=0):
    rng = np.random.default_rng(seed)
    return xr.Dataset(
        {"CMI": (("y", "x"), 250 + rng.normal(0, 0.1, (ny, nx)))},
        coords={"y": np.arange(ny, 0, -1.0), "x": np.arange(nx, dtype=float)})


def test_cache_roundtrip(tmp_path):
    cfg = StereoConfig(goes_cache=str(tmp_path))
    when = dt.datetime(2020, 6, 1, 12, 0)
    ds = _fake_ds()
    ds.attrs["cstereo_domain"] = "C"
    # write into the cache the way fetch_channel would, then read via it
    from contrailstereo.data.goes import _cache_path
    ds.to_netcdf(_cache_path(cfg, 16, 14, when))
    out = fetch_channel(16, 14, when, cfg)          # must hit disk, no network
    np.testing.assert_allclose(out.CMI.values, ds.CMI.values)
    assert out.attrs["cstereo_domain"] == "C"


def test_flag_rows_finds_injected_stripes():
    ds14, ds15 = _fake_ds(seed=1), _fake_ds(seed=2)
    for r in (10, 11, 25):                          # inject noisy rows in BTD
        ds14["CMI"].values[r] += np.random.default_rng(r).normal(0, 1.5, 60)
    bad = flag_rows(row_artifact_scores(ds14, ds15))
    for r in (10, 11, 25):
        assert r in bad
    assert len(bad) < 12                            # dilation only, no spray


def test_flag_rows_clean_field_flags_nothing():
    bad = flag_rows(row_artifact_scores(_fake_ds(seed=3), _fake_ds(seed=4)))
    assert bad.size <= 2


def test_masked_copy_nans_rows_and_preserves_rest():
    ds = _fake_ds()
    out = masked_copy(ds, np.array([5, 6]))
    assert np.isnan(out.CMI.values[5]).all()
    assert np.isfinite(out.CMI.values[7]).all()
    assert np.isfinite(ds.CMI.values[5]).all()       # original untouched


@pytest.mark.network
def test_live_fetch_one_scene():
    fields, domain = __import__(
        "contrailstereo.data.goes", fromlist=["fetch_pair"]
    ).fetch_pair(dt.datetime(2021, 4, 14, 10, 6), 16)
    assert 13 in fields and 15 in fields and domain in ("C", "F")