"""GOES imagery: cached fetch + detector-row striping conditioning."""

from __future__ import annotations

import os
import numpy as np
import xarray as xr

from ..config import StereoConfig, DEFAULT


# ------------------------------------------------------------------ cache
def _cache_path(cfg, sat, ch, when):
    os.makedirs(cfg.goes_cache, exist_ok=True)
    return os.path.join(cfg.goes_cache,
                        f"G{sat}_C{ch:02d}_{when:%Y%m%d%H%M}.nc")


def _download(sat, ch, when):
    """Live fetch via goes2go: CONUS, falling back to full disk."""
    from goes2go import GOES                      # lazy, not needed offline
    last = None
    for domain in ("C", "F"):
        try:
            ds = GOES(satellite=sat, product="ABI-L2-CMIP", domain=domain,
                      channel=ch).nearesttime(when, return_as="xarray",
                                              download=True, verbose=False)
            ds.attrs["cstereo_domain"] = domain
            return ds
        except Exception as e:                    # noqa: BLE001
            last = e
    raise last


def _plain_encoding(ds):
    for name, var in ds.variables.items():
        if (np.issubdtype(var.dtype, np.datetime64)
                or np.issubdtype(var.dtype, np.timedelta64)):
            continue
        var.encoding = {}
    return ds


def fetch_channel(sat, ch, when, cfg: StereoConfig = DEFAULT):
    p = _cache_path(cfg, sat, ch, when)
    if os.path.exists(p):
        return xr.open_dataset(p)
    ds = _plain_encoding(_download(sat, ch, when))
    ds.to_netcdf(p, encoding={"CMI": {"dtype": "float32",
                                      "zlib": True, "complevel": 4}})
    return ds


def fetch_pair(when, sat, cfg: StereoConfig = DEFAULT, channels=None):
    """Fetch a pair of channels for one satellite 
    (cfg.match_channels unless overridden).

    Returns
    -------
    fields : {channels[0]: Dataset, channels[1]: Dataset}
    domain : "C" or "F" (whichever the C14 fetch used)
    """
    channels = channels or cfg.match_channels
    fields = {ch: fetch_channel(sat, ch, when, cfg) for ch in channels}
    return fields, fields[channels[0]].attrs.get("cstereo_domain", "?")


# Striping for G17
def row_artifact_scores(ds14, ds15):
    """Per-detector-row artifact score on the full-sector BTD: mean
    |row-Laplacian| normalised by the sector median. Magnitude-based so
    bipolar striping cannot cancel."""
    btd = np.asarray(ds14["CMI"].values, float) - \
          np.asarray(ds15["CMI"].values, float)
    lap = btd[1:-1] - 0.5 * (btd[:-2] + btd[2:])
    e = np.nanmean(np.abs(lap), axis=1)
    e = np.pad(e, 1, constant_values=np.nan)
    return e / np.nanmedian(e)


def flag_rows(score, cfg: StereoConfig = DEFAULT):
    """Rows exceeding a robust MAD z-threshold, dilated +-cfg.stripe_dilate.
    Known limitation (documented, scene 147): broadband LHP noise makes the
    discrete-row model over-fire; callers can gate on flagged fraction."""
    nsig, dilate = cfg.stripe_nsig, int(cfg.stripe_dilate)
    med = np.nanmedian(score)
    mad = np.nanmedian(np.abs(score - med))
    bad = np.where((score - med) / (1.4826 * mad + 1e-12) > nsig)[0]
    if bad.size and dilate:
        bad = np.unique(np.concatenate(
            [bad + d for d in range(-dilate, dilate + 1)]))
        bad = bad[(bad >= 0) & (bad < score.size)]
    return bad.astype(int)


def masked_copy(ds, bad_rows):
    """Dataset copy with flagged native rows NaN'd (applied to BOTH
    channels with the same rows so the BTD stays consistent)."""
    out = ds.copy(deep=True)
    cmi = out["CMI"].values.copy()
    cmi[np.asarray(bad_rows, int), :] = np.nan
    out["CMI"].values = cmi
    return out