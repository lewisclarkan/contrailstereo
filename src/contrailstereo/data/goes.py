"""GOES imagery: cached fetch + detector-row striping conditioning."""

from __future__ import annotations

import logging
import os

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

log = logging.getLogger(__name__)

DOMAIN_ATTR = "cstereo_domain"      # "C = CONUS"
SAME_SCAN_TOL_S = 60.0



# ---------- Fetch ----------

def cache_path(cache_dir, sat, ch, when) -> Path:
    when = pd.Timestamp(when)
    return Path(cache_dir) / f"G{sat}_C{ch:02d}_{when:%Y%m%d%H%M}.nc"


def _download(sat, ch, when) -> xr.Dataset:
    """Live fetch via goes2go: CONUS, falling back to full disk."""
    from goes2go import GOES
    last = None
    for domain in ("C", "F"):
        try:
            ds = GOES(satellite=sat, product="ABI-L2-CMIP", domain=domain,
                      channel=ch).nearesttime(
                pd.Timestamp(when).to_pydatetime(), return_as="xarray",
                download=True, verbose=False)
            ds.attrs[DOMAIN_ATTR] = domain
            return ds
        except Exception as e:                   
            log.info("G%s C%02d domain %s unavailable: %s", sat, ch, domain, e)
            last = e
    raise last


def _plain_encoding(ds):
    for var in ds.variables.values():
        if not (np.issubdtype(var.dtype, np.datetime64)
                or np.issubdtype(var.dtype, np.timedelta64)):
            var.encoding = {}
    return ds


def fetch_channel(sat, ch, when, cache_dir) -> xr.Dataset:
    """One ABI L2 CMIP channel nearest to `when`, cache-first"""
    p = cache_path(cache_dir, sat, ch, when)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        ds = _plain_encoding(_download(sat, ch, when))
        tmp = p.with_suffix(".nc.part")
        ds.to_netcdf(tmp, encoding={"CMI": {"dtype": "float32",
                                            "zlib": True, "complevel": 4}})
        os.replace(tmp, p)
    return xr.open_dataset(p)


def scan_time(ds) -> pd.Timestamp:
    """Nominal scan time of a CMIP dataset (UTC, tz-naive)."""
    return pd.Timestamp(ds["t"].values)


def fetch_channels(sat, channels, when, cache_dir):
    """Several channels of one satellite for the same scan.
 
    Returns ({ch: Dataset}, domain). Raises if the channels came from
    different domains or scans -- their difference would not be a BTD.
    """
    fields = {ch: fetch_channel(sat, ch, when, cache_dir) for ch in channels}
    doms = {ch: ds.attrs.get(DOMAIN_ATTR, "?") for ch, ds in fields.items()}
    times = {ch: scan_time(ds) for ch, ds in fields.items()}
    spread = (max(times.values()) - min(times.values())).total_seconds()
    if len(set(doms.values())) > 1 or spread > SAME_SCAN_TOL_S:
        raise ValueError(f"G{sat} channels not from one scan: "
                         f"domains {doms}, times {times}")
    return fields, next(iter(doms.values()))


# ---------- G17 destriping (experimental) ----------


def row_scores(ds_a, ds_b) -> np.ndarray:
    """Per-detector-row artefact score on the full-sector BTD: mean
    |row Laplacian| normalised by its sector median. Magnitude-based, so
    bipolar striping cannot cancel."""
    btd = (np.asarray(ds_a["CMI"].values, float)
           - np.asarray(ds_b["CMI"].values, float))
    lap = btd[1:-1] - 0.5 * (btd[:-2] + btd[2:])
    e = np.nanmean(np.abs(lap), axis=1)
    e = np.pad(e, 1, constant_values=np.nan)
    return e / np.nanmedian(e)
 
 
def flag_rows(scores, nsig=4.0, dilate=1) -> np.ndarray:
    """Rows above a robust (MAD) z-threshold, dilated by +-dilate rows.
 
    Known limitation: broadband noise (e.g. the G17 loop-heat-pipe period)
    breaks the discrete-row model and over-flags; gate on the flagged
    fraction if that matters.
    """
    med = np.nanmedian(scores)
    mad = np.nanmedian(np.abs(scores - med))
    bad = np.where((scores - med) / (1.4826 * mad + 1e-12) > nsig)[0]
    if bad.size and dilate:
        bad = np.unique(np.concatenate(
            [bad + d for d in range(-int(dilate), int(dilate) + 1)]))
        bad = bad[(bad >= 0) & (bad < scores.size)]
    return bad.astype(int)
 
 
def stripe_rows(ds_a, ds_b, nsig=4.0, dilate=1):
    """Flag striped rows on a channel pair. Returns (rows, scores)."""
    scores = row_scores(ds_a, ds_b)
    return flag_rows(scores, nsig, dilate), scores
 
 
def mask_rows(ds, rows) -> xr.Dataset:
    """Copy with the given native rows set to NaN. Apply the same rows to
    both channels so the BTD stays consistent."""
    out = ds.copy(deep=True)
    cmi = out["CMI"].values.copy()
    cmi[np.asarray(rows, int), :] = np.nan
    out["CMI"].values = cmi
    return out